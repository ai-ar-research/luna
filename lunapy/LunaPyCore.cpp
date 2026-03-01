// LirpaPyCore.cpp - pybind11 bindings for LIRPA C++ backend

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include "engine/TorchModel.h"
#include "common/BoundedTensor.h"
#include "configuration/LunaConfiguration.h"
#include "MString.h"

namespace py = pybind11;

// Helper function to convert torch::Tensor to NumPy array
py::array_t<float> torch_to_numpy(const torch::Tensor& tensor) {
    // Ensure tensor is contiguous and on CPU
    torch::Tensor cpu_tensor = tensor.contiguous().cpu();
    
    // Get tensor dimensions
    std::vector<ssize_t> shape;
    for (int64_t i = 0; i < cpu_tensor.dim(); ++i) {
        shape.push_back(cpu_tensor.size(i));
    }
    
    // Calculate total number of elements
    ssize_t size = 1;
    for (auto s : shape) {
        size *= s;
    }
    
    // Create numpy array and copy data (safer than sharing)
    py::array_t<float> result(shape);
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    
    // Copy data from tensor to numpy
    std::memcpy(ptr, cpu_tensor.data_ptr<float>(), size * sizeof(float));
    
    return result;
}

// Helper function to convert NumPy array to torch::Tensor
torch::Tensor numpy_to_torch(py::array_t<float> array) {
    py::buffer_info buf = array.request();
    
    // Get shape
    std::vector<int64_t> shape;
    for (ssize_t i = 0; i < buf.ndim; ++i) {
        shape.push_back(buf.shape[i]);
    }
    
    // Create tensor from data (makes a copy)
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor = torch::from_blob(
        buf.ptr,
        shape,
        options
    ).clone();

    return tensor.to(LunaConfiguration::getDevice());
}

// Python wrapper for BoundedTensor
class PyBoundedTensor {
public:
    BoundedTensor<torch::Tensor> cpp_bounded_tensor;
    
    PyBoundedTensor() {}
    
    PyBoundedTensor(py::array_t<float> lower, py::array_t<float> upper) {
        torch::Tensor lower_tensor = numpy_to_torch(lower);
        torch::Tensor upper_tensor = numpy_to_torch(upper);
        cpp_bounded_tensor = BoundedTensor<torch::Tensor>(lower_tensor, upper_tensor);
    }
    
    PyBoundedTensor(const BoundedTensor<torch::Tensor>& bt) 
        : cpp_bounded_tensor(bt) {}
    
    py::array_t<float> lower() const {
        return torch_to_numpy(cpp_bounded_tensor.lower());
    }
    
    py::array_t<float> upper() const {
        return torch_to_numpy(cpp_bounded_tensor.upper());
    }
    
    const BoundedTensor<torch::Tensor>& get_cpp_object() const {
        return cpp_bounded_tensor;
    }
};

// Python wrapper for TorchModel
class PyTorchModel {
private:
    std::shared_ptr<NLR::TorchModel> model;
    
public:
    // Constructor: ONNX only
    PyTorchModel(const std::string& onnx_path) {
        model = std::make_shared<NLR::TorchModel>(String(onnx_path.c_str()));
    }
    
    // Constructor: ONNX + VNN-LIB
    PyTorchModel(const std::string& onnx_path, const std::string& vnnlib_path) {
        model = std::make_shared<NLR::TorchModel>(
            String(onnx_path.c_str()),
            String(vnnlib_path.c_str())
        );
    }
    
    // Get model info
    unsigned getInputSize() const { return model->getInputSize(); }
    unsigned getOutputSize() const { return model->getOutputSize(); }
    unsigned getNumNodes() const { return model->getNumNodes(); }
    
    // Set input bounds from NumPy arrays
    void setInputBounds(py::array_t<float> lower, py::array_t<float> upper) {
        torch::Tensor lower_tensor = numpy_to_torch(lower);
        torch::Tensor upper_tensor = numpy_to_torch(upper);
        BoundedTensor<torch::Tensor> bounds(lower_tensor, upper_tensor);
        model->setInputBounds(bounds);
    }
    
    // Get input bounds
    PyBoundedTensor getInputBounds() const {
        BoundedTensor<torch::Tensor> bounds = model->getInputBounds();
        return PyBoundedTensor(bounds);
    }
    
    bool hasInputBounds() const {
        return model->hasInputBounds();
    }
    
    // Forward pass
    py::array_t<float> forward(py::array_t<float> input) {
        torch::Tensor input_tensor = numpy_to_torch(input);
        auto activations = model->forwardAndStoreActivations(input_tensor);
        unsigned output_idx = model->getOutputIndex();
        torch::Tensor output = activations[output_idx];
        return torch_to_numpy(output);
    }
    
    // Set specification matrix
    void setSpecificationMatrix(py::array_t<float> spec_matrix) {
        torch::Tensor spec_tensor = numpy_to_torch(spec_matrix);
        
        // Ensure spec matrix is 3D: [num_specs, batch, output_size]
        // If 2D [num_specs, output_size], add batch dimension
        if (spec_tensor.dim() == 2) {
            spec_tensor = spec_tensor.unsqueeze(1); // [num_specs, 1, output_size]
        } else if (spec_tensor.dim() != 3) {
            throw std::invalid_argument("Specification matrix must be 2D or 3D");
        }
        
        model->setSpecificationMatrix(spec_tensor);
    }
    
    bool hasSpecificationMatrix() const {
        return model->hasSpecificationMatrix();
    }
    
    // Compute bounds - main entry point
    PyBoundedTensor compute_bounds(
        const std::string& method = "CROWN",
        bool bound_lower = true,
        bool bound_upper = true,
        py::object C = py::none()
    ) {
        // Parse method string
        LunaConfiguration::AnalysisMethod analysis_method;
        if (method == "CROWN" || method == "crown") {
            analysis_method = LunaConfiguration::AnalysisMethod::CROWN;
        } else if (method == "alpha-CROWN" || method == "AlphaCROWN" || method == "alphacrown") {
            analysis_method = LunaConfiguration::AnalysisMethod::AlphaCROWN;
        } else {
            throw std::invalid_argument("Unknown method: " + method + ". Use 'CROWN' or 'alpha-CROWN'");
        }
        
        // Set specification matrix if provided
        torch::Tensor* spec_ptr = nullptr;
        torch::Tensor spec_tensor;
        if (!C.is_none()) {
            py::array_t<float> C_array = C.cast<py::array_t<float>>();
            spec_tensor = numpy_to_torch(C_array);
            
            // Ensure C is 3D: [num_specs, batch, output_size]
            // If 2D [num_specs, output_size], add batch dimension
            if (spec_tensor.dim() == 2) {
                spec_tensor = spec_tensor.unsqueeze(1); // [num_specs, 1, output_size]
            } else if (spec_tensor.dim() != 3) {
                throw std::invalid_argument("Specification matrix C must be 2D or 3D");
            }
            
            spec_ptr = &spec_tensor;
        }
        
        // Get input bounds
        if (!model->hasInputBounds()) {
            throw std::runtime_error("Input bounds not set. Call setInputBounds() or provide vnnlib_path in constructor.");
        }
        BoundedTensor<torch::Tensor> input_bounds = model->getInputBounds();
        
        // Call compute_bounds
        BoundedTensor<torch::Tensor> result = model->compute_bounds(
            input_bounds,
            spec_ptr,
            analysis_method,
            bound_lower,
            bound_upper
        );
        
        return PyBoundedTensor(result);
    }
    
    // Direct analysis methods
    PyBoundedTensor runCROWN() {
        BoundedTensor<torch::Tensor> result = model->runCROWN();
        return PyBoundedTensor(result);
    }
    
    PyBoundedTensor runAlphaCROWN(bool optimizeLower = true, bool optimizeUpper = false) {
        BoundedTensor<torch::Tensor> result = model->runAlphaCROWN(optimizeLower, optimizeUpper);
        return PyBoundedTensor(result);
    }
    
    // Get final analysis bounds
    PyBoundedTensor getFinalAnalysisBounds() const {
        if (!model->hasFinalAnalysisBounds()) {
            throw std::runtime_error("No final analysis bounds available. Run compute_bounds() first.");
        }
        BoundedTensor<torch::Tensor> bounds = model->getFinalAnalysisBounds();
        return PyBoundedTensor(bounds);
    }
    
    bool hasFinalAnalysisBounds() const {
        return model->hasFinalAnalysisBounds();
    }
};

// pybind11 module definition
PYBIND11_MODULE(LirpaPyCore, m) {
    m.doc() = "LIRPA Python bindings - Neural network bound propagation";
    
    // BoundedTensor class
    py::class_<PyBoundedTensor>(m, "BoundedTensor")
        .def(py::init<>())
        .def(py::init<py::array_t<float>, py::array_t<float>>(),
             py::arg("lower"), py::arg("upper"),
             "Create a BoundedTensor from lower and upper bound arrays")
        .def("lower", &PyBoundedTensor::lower,
             "Get lower bounds as NumPy array")
        .def("upper", &PyBoundedTensor::upper,
             "Get upper bounds as NumPy array")
        .def("__repr__", [](const PyBoundedTensor& bt) {
            return "<BoundedTensor with lower/upper bounds>";
        });
    
    // TorchModel class
    py::class_<PyTorchModel>(m, "TorchModel")
        .def(py::init<const std::string&>(),
             py::arg("onnx_path"),
             "Create TorchModel from ONNX file")
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("onnx_path"), py::arg("vnnlib_path"),
             "Create TorchModel from ONNX and VNN-LIB files")
        .def("getInputSize", &PyTorchModel::getInputSize,
             "Get input size")
        .def("getOutputSize", &PyTorchModel::getOutputSize,
             "Get output size")
        .def("getNumNodes", &PyTorchModel::getNumNodes,
             "Get number of nodes in the model")
        .def("setInputBounds", &PyTorchModel::setInputBounds,
             py::arg("lower"), py::arg("upper"),
             "Set input bounds from lower and upper NumPy arrays")
        .def("getInputBounds", &PyTorchModel::getInputBounds,
             "Get input bounds as BoundedTensor")
        .def("hasInputBounds", &PyTorchModel::hasInputBounds,
             "Check if input bounds are set")
        .def("forward", &PyTorchModel::forward,
             py::arg("input"),
             "Forward pass through the model")
        .def("setSpecificationMatrix", &PyTorchModel::setSpecificationMatrix,
             py::arg("C"),
             "Set specification matrix for output constraints")
        .def("hasSpecificationMatrix", &PyTorchModel::hasSpecificationMatrix,
             "Check if specification matrix is set")
        .def("compute_bounds", &PyTorchModel::compute_bounds,
             py::arg("method") = "CROWN",
             py::arg("bound_lower") = true,
             py::arg("bound_upper") = true,
             py::arg("C") = py::none(),
             "Compute output bounds using specified method (CROWN or alpha-CROWN)")
        .def("runCROWN", &PyTorchModel::runCROWN,
             "Run CROWN analysis")
        .def("runAlphaCROWN", &PyTorchModel::runAlphaCROWN,
             py::arg("optimizeLower") = true,
             py::arg("optimizeUpper") = false,
             "Run Alpha-CROWN analysis")
        .def("getFinalAnalysisBounds", &PyTorchModel::getFinalAnalysisBounds,
             "Get final analysis bounds from last computation")
        .def("hasFinalAnalysisBounds", &PyTorchModel::hasFinalAnalysisBounds,
             "Check if final analysis bounds are available")
        .def("__repr__", [](const PyTorchModel& model) {
            return "<TorchModel with " + std::to_string(model.getNumNodes()) + 
                   " nodes, input_size=" + std::to_string(model.getInputSize()) +
                   ", output_size=" + std::to_string(model.getOutputSize()) + ">";
        });
    
    // Expose LunaConfiguration settings
    py::class_<LunaConfiguration>(m, "LunaConfiguration")
        .def_readwrite_static("VERBOSE", &LunaConfiguration::VERBOSE)
        .def_readwrite_static("COMPUTE_LOWER", &LunaConfiguration::COMPUTE_LOWER)
        .def_readwrite_static("COMPUTE_UPPER", &LunaConfiguration::COMPUTE_UPPER)
        .def_readwrite_static("ALPHA_ITERATIONS", &LunaConfiguration::ALPHA_ITERATIONS)
        .def_readwrite_static("ALPHA_LR", &LunaConfiguration::ALPHA_LR)
        .def_readwrite_static("OPTIMIZE_LOWER", &LunaConfiguration::OPTIMIZE_LOWER)
        .def_readwrite_static("OPTIMIZE_UPPER", &LunaConfiguration::OPTIMIZE_UPPER)
        .def_readwrite_static("STABILIZE_INTERMEDIATE_BOUNDS", &LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS)
        .def_readwrite_static("USE_CUDA", &LunaConfiguration::USE_CUDA)
        .def_readwrite_static("CUDA_DEVICE_ID", &LunaConfiguration::CUDA_DEVICE_ID)
        .def_static("set_device", [](const std::string& device_str) {
            if (device_str == "cpu" || device_str == "CPU") {
                LunaConfiguration::USE_CUDA = false;
            } else if (device_str.rfind("cuda", 0) == 0 || device_str.rfind("CUDA", 0) == 0) {
                LunaConfiguration::USE_CUDA = true;
                auto colon = device_str.find(':');
                if (colon != std::string::npos) {
                    LunaConfiguration::CUDA_DEVICE_ID = std::stoi(device_str.substr(colon + 1));
                }
            }
            LunaConfiguration::updateDeviceFromFlags();
        });
    
    // Version info
    m.attr("__version__") = LIRPA_VERSION;
}
