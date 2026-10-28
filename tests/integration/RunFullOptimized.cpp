#include "src/input_parsers/OnnxToTorch.h"
#include "src/engine/TorchModel.h"
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/configuration/LunaConfiguration.h"
#include "src/common/CommonError.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <thread>
#include <algorithm>

// Performance timer utility
class Timer
{
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    std::string name;

public:
    Timer( const std::string &n = "" )
        : start( clock::now() )
        , name( n )
    {
    }
    ~Timer()
    {
        if ( !name.empty() )
        {
            double elapsed = std::chrono::duration<double>( clock::now() - start ).count();
            std::cout << name << " took " << std::fixed << std::setprecision( 3 ) << elapsed
                      << " seconds" << std::endl;
        }
    }
    double elapsed()
    {
        return std::chrono::duration<double>( clock::now() - start ).count();
    }
};

// Helper function to print bounds with optional rounding
static void printBounds( const torch::Tensor &lower, const torch::Tensor &upper, bool round = true )
{
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if ( lb.dim() == 0 )
        lb = lb.unsqueeze( 0 );
    if ( ub.dim() == 0 )
        ub = ub.unsqueeze( 0 );

    std::cout << "Output Bounds:" << std::endl;

    if ( round )
    {
        std::cout << std::fixed << std::setprecision( 6 );
    }
    else
    {
        std::cout << std::scientific << std::setprecision( 15 );
    }

    int numElements = std::min( 10, static_cast<int>( lb.numel() ) );
    for ( int i = 0; i < numElements; ++i )
    {
        if ( i > 0 )
            std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if ( l.dim() > 0 )
            l = l.flatten()[0];
        if ( u.dim() > 0 )
            u = u.flatten()[0];
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
    std::cout << std::defaultfloat;
}

int main( int argc, char *argv[] )
{
    // Parse command-line arguments
    LunaConfiguration::parseArgs( argc, argv );

    try
    {
        // OPTIMIZATION 1: Use optimal thread count instead of 1
        int num_threads = std::max( 1, (int)std::thread::hardware_concurrency() / 2 );

        // For deterministic debugging, you can still force single-threaded
        bool deterministic = true; // Set to true for debugging
        if ( deterministic )
        {
            num_threads = 4;
            std::cout << "Running in deterministic single-threaded mode for debugging\n";
        }

        at::set_num_threads( num_threads );
        at::set_num_interop_threads( num_threads );
        std::cout << "Set threads to " << num_threads << " for optimal performance\n\n";

// OPTIMIZATION 2: Set environment variables for MKL/BLAS optimization
#ifdef USE_MKL
        // If using Intel MKL
        setenv( "MKL_NUM_THREADS", std::to_string( num_threads ).c_str(), 1 );
        setenv( "MKL_DYNAMIC", "FALSE", 1 );
#endif

        // Configuration flags
        bool round = true;

        // File paths (overridable from CLI)
        std::string onnxFilePath = "../resources/onnx/cifar_base_kw_simp.onnx";
        std::string vnnlibFilePath = "../resources/onnx/vnnlib/cifar_bounded.vnnlib";
        if ( argc >= 3 )
        {
            onnxFilePath = argv[1];
            vnnlibFilePath = argv[2];
        }
        // unsigned iterations = 20;

        std::cout << "Parsing ONNX file: " << onnxFilePath << std::endl;
        std::cout << "Parsing VNN-LIB file: " << vnnlibFilePath << std::endl;

        // Time model loading
        std::shared_ptr<NLR::TorchModel> torchModel;
        {
            Timer timer( "Model loading" );
            torchModel = std::make_shared<NLR::TorchModel>( String( onnxFilePath.c_str() ),
                                                            String( vnnlibFilePath.c_str() ) );
        }

        std::cout << "TorchModel created successfully" << std::endl;
        std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
        std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
        std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;

        // Print network structure (only in verbose mode)
        bool verbose = false;
        if ( verbose )
        {
            std::cout << "\nNetwork Structure" << std::endl;
            for ( unsigned i = 0; i < torchModel->getNumNodes(); ++i )
            {
                auto node = torchModel->getNode( i );
                if ( !node )
                    continue;

                std::cout << "Node " << i << ": ";
                switch ( node->getNodeType() )
                {
                case NLR::NodeType::INPUT:
                    std::cout << "INPUT";
                    break;
                case NLR::NodeType::LINEAR:
                    std::cout << "LINEAR";
                    break;
                case NLR::NodeType::RELU:
                    std::cout << "RELU";
                    break;
                case NLR::NodeType::CONSTANT:
                    std::cout << "CONSTANT";
                    break;
                case NLR::NodeType::IDENTITY:
                    std::cout << "IDENTITY";
                    break;
                case NLR::NodeType::RESHAPE:
                    std::cout << "RESHAPE";
                    break;
                case NLR::NodeType::FLATTEN:
                    std::cout << "FLATTEN";
                    break;
                case NLR::NodeType::SUB:
                    std::cout << "SUB";
                    break;
                case NLR::NodeType::ADD:
                    std::cout << "ADD";
                    break;
                case NLR::NodeType::CONV:
                    std::cout << "CONV";
                    break;
                default:
                    std::cout << "UNKNOWN";
                }
                std::cout << " (in=" << node->getInputSize() << ", out=" << node->getOutputSize()
                          << ")";

                auto deps = torchModel->getDependencies( i );
                if ( !deps.empty() )
                {
                    std::cout << " <- deps: [";
                    for ( size_t j = 0; j < deps.size(); ++j )
                    {
                        std::cout << deps[j];
                        if ( j < deps.size() - 1 )
                            std::cout << ", ";
                    }
                    std::cout << "]";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        // Verify input bounds
        if ( !torchModel->hasInputBounds() )
        {
            std::cerr << "ERROR: Input bounds not loaded from VNN-LIB file!" << std::endl;
            return 1;
        }

        BoundedTensor<torch::Tensor> inputBounds = torchModel->getInputBounds();
        std::cout << "\nInput bounds loaded from VNN-LIB file:" << std::endl;
        torch::Tensor lowerBounds = inputBounds.lower();
        torch::Tensor upperBounds = inputBounds.upper();

        // OPTIMIZATION 4: Avoid unnecessary dtype conversions
        if ( lowerBounds.dtype() != torch::kFloat32 )
        {
            lowerBounds = lowerBounds.to( torch::kFloat32 );
            upperBounds = upperBounds.to( torch::kFloat32 );
            inputBounds = BoundedTensor<torch::Tensor>( lowerBounds, upperBounds );
            std::cout << "Converted input bounds to float32" << std::endl;
        }

        std::cout << "  Input bounds: " << lowerBounds.size( 0 ) << " variables bounded in ["
                  << lowerBounds.min().item<double>() << ", " << upperBounds.max().item<double>()
                  << "]" << std::endl;

        // Debug: compute model output at the VNNLIB box center and print first 10 values.
        // This should ALWAYS lie inside any valid output bounds over the same box.
        {
            torch::NoGradGuard no_grad;
            torch::Tensor center = ( lowerBounds + upperBounds ) / 2.0;

            // TorchModel forward expects ONNX-declared input shape.
            // For this TinyImageNet model we observed 9408 = 3*56*56.
            if ( center.numel() == 9408 )
            {
                center = center.view( { 1, 3, 56, 56 } );
            }
            else if ( center.numel() == 12288 )
            {
                center = center.view( { 1, 3, 64, 64 } );
            }
            else if ( center.numel() == 3072 )
            {
                center = center.view( { 1, 3, 32, 32 } );
            }
            else if ( center.numel() == 784 )
            {
                center = center.view( { 1, 1, 28, 28 } );
            }
            else
            {
                center = center.view( { 1, (long)center.numel() } );
            }

            auto acts = torchModel->forwardAndStoreActivations( center );
            torch::Tensor y = acts[torchModel->getOutputIndex()];
            y = y.flatten().to( torch::kFloat32 );

            std::cout << "\nCenter-point forward output (first 10):" << std::endl;
            std::cout << std::fixed << std::setprecision( 6 );
            int nprint = std::min<int>( (int)y.numel(), 10 );
            for ( int i = 0; i < nprint; ++i )
            {
                std::cout << "  y[" << i << "]=" << y[i].item<float>() << std::endl;
            }
            std::cout << "  y.min=" << y.min().item<float>() << " y.max=" << y.max().item<float>()
                      << std::endl;
            std::cout << std::defaultfloat;
        }

        // Run CROWN analysis with timing
        std::cout << "\nRunning plain CROWN analysis..." << std::endl;
        BoundedTensor<torch::Tensor> crownResult;
        {
            Timer timer( "CROWN analysis" );
            crownResult = torchModel->compute_bounds(
                inputBounds, nullptr, LunaConfiguration::AnalysisMethod::CROWN, true, true );
        }

        std::cout << "\n=== CROWN Results ===" << std::endl;
        if ( crownResult.lower().defined() && crownResult.upper().defined() )
        {
            printBounds( crownResult.lower(), crownResult.upper(), round );

            // Soundness sanity check: center-point output must lie inside [lb, ub] for each output.
            // This catches cases where backward bound propagation is inconsistent with the forward
            // model.
            {
                torch::NoGradGuard no_grad;
                torch::Tensor center = ( lowerBounds + upperBounds ) / 2.0;
                if ( center.numel() == 9408 )
                {
                    center = center.view( { 1, 3, 56, 56 } );
                }
                else if ( center.numel() == 12288 )
                {
                    center = center.view( { 1, 3, 64, 64 } );
                }
                else if ( center.numel() == 3072 )
                {
                    center = center.view( { 1, 3, 32, 32 } );
                }
                else if ( center.numel() == 784 )
                {
                    center = center.view( { 1, 1, 28, 28 } );
                }
                else
                {
                    center = center.view( { 1, (long)center.numel() } );
                }

                auto acts = torchModel->forwardAndStoreActivations( center );
                torch::Tensor y =
                    acts[torchModel->getOutputIndex()].flatten().to( torch::kFloat32 );
                torch::Tensor lb = crownResult.lower().flatten().to( torch::kFloat32 );
                torch::Tensor ub = crownResult.upper().flatten().to( torch::kFloat32 );

                int64_t n =
                    std::min<int64_t>( y.numel(), std::min<int64_t>( lb.numel(), ub.numel() ) );
                int64_t bad = 0;
                for ( int64_t i = 0; i < n; ++i )
                {
                    float yi = y[i].item<float>();
                    float lbi = lb[i].item<float>();
                    float ubi = ub[i].item<float>();
                    if ( !( lbi <= yi && yi <= ubi ) )
                        bad++;
                }
                std::cout << "\nCenter containment check: " << ( n - bad ) << "/" << n
                          << " outputs contained" << std::endl;
                if ( bad > 0 )
                {
                    std::cout << "WARNING: bounds are not sound for at least one output under the "
                                 "center point"
                              << std::endl;
                }
            }
        }

        return 0;
    }
    catch ( const CommonError &e )
    {
        std::cerr << "CommonError: code=" << e.getCode();
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            std::cerr << " message=" << msg;
        std::cerr << std::endl;
        return 1;
    }
    catch ( const Error &e )
    {
        std::cerr << "Error: class=" << e.getErrorClass() << " code=" << e.getCode();
        const char *msg = e.getUserMessage();
        if ( msg && msg[0] != '\0' )
            std::cerr << " message=" << msg;
        std::cerr << std::endl;
        return 1;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
