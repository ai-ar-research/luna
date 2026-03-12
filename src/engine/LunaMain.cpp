#include "LunaMain.h"
#include "input_parsers/OnnxToTorch.h"
#include "TorchModel.h"
#include "LunaError.h"
#include "configuration/LunaConfiguration.h"
#include "input_parsers/OutputConstraint.h"
#include "input_parsers/VnnLibInputParser.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

// Helper function to print bounds
static void printBounds(const torch::Tensor& lower, const torch::Tensor& upper) {
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if (lb.dim() == 0) lb = lb.unsqueeze(0);
    if (ub.dim() == 0) ub = ub.unsqueeze(0);

    std::cout << std::fixed << std::setprecision(6);

    /*
    // Paired format: [lower, upper] per output
    std::cout << "Output Bounds:" << std::endl;
    for (int i = 0; i < lb.size(0); ++i) {
        if (i > 0) std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if (l.dim() > 0) l = l.flatten()[0];
        if (u.dim() > 0) u = u.flatten()[0];
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
    */
    std::cout << " " << std::endl;
    // Separate lower/upper lists
    std::cout << "Lower Bounds: [";
    for (int i = 0; i < lb.size(0); ++i) {
        auto l = lb[i];
        if (l.dim() > 0) l = l.flatten()[0];
        if (i > 0) std::cout << ", ";
        std::cout << l.item<double>();
    }
    std::cout << "]" << std::endl;

    std::cout << "Upper Bounds: [";
    for (int i = 0; i < ub.size(0); ++i) {
        auto u = ub[i];
        if (u.dim() > 0) u = u.flatten()[0];
        if (i > 0) std::cout << ", ";
        std::cout << u.item<double>();
    }
    std::cout << "]" << std::endl;

    std::cout << std::defaultfloat;
}

enum class PropertyStatus {
    Unsat,   // safe/verified - NO input violates the property
    Sat,     // unsafe - counterexample exists
    Unknown  // inconclusive
};

static PropertyStatus evaluatePropertyStatus(
    const NLR::OutputConstraintSet& constraints,
    const torch::Tensor& lowerBounds,
    const torch::Tensor& upperBounds,
    std::string& detail) {
    if (!constraints.hasConstraints()) {
        detail = "no output constraints found in VNN-LIB";
        return PropertyStatus::Unknown;
    }

    if (!lowerBounds.defined() || !upperBounds.defined()) {
        detail = "bounds are undefined";
        return PropertyStatus::Unknown;
    }

    torch::Tensor lb = lowerBounds.flatten();
    torch::Tensor ub = upperBounds.flatten();

    if (lb.numel() == 0 || ub.numel() == 0) {
        detail = "bounds are empty";
        return PropertyStatus::Unknown;
    }

    NLR::CMatrixResult cMatrix = constraints.toCMatrix();
    torch::Tensor thresholds = cMatrix.thresholds.to(ub.device());

    if (lb.numel() != thresholds.numel() || ub.numel() != thresholds.numel()) {
        detail = "bounds/threshold size mismatch";
        return PropertyStatus::Unknown;
    }

    if (cMatrix.hasORBranches) {
        Vector<NLR::BranchResult> branchResults =
            NLR::OutputConstraintSet::evaluateORBranches(lb, ub, thresholds,
                                                         cMatrix.branchMapping,
                                                         cMatrix.branchSizes);
        // OR of counterexample branches: unsafe if ANY branch is satisfied.
        // Unsat (safe): ALL branches must be disproved (each branch's AND broken).
        // Sat (unsafe): ANY branch is always satisfiable.
        bool allBranchesDisproved = true;
        bool anyBranchAlwaysSat = false;

        for (const auto& branch : branchResults) {
            if (!branch.verified) {
                allBranchesDisproved = false;
            }
            if (branch.refuted) {
                anyBranchAlwaysSat = true;
            }
        }

        if (allBranchesDisproved) {
            detail = "all OR-branches disproved (no counterexample possible)";
            return PropertyStatus::Unsat;
        }
        if (anyBranchAlwaysSat) {
            detail = "at least one OR-branch always satisfiable (counterexample exists)";
            return PropertyStatus::Sat;
        }
        detail = "bounds inconclusive for OR-branches";
        return PropertyStatus::Unknown;
    }

    // C @ y <= rhs is the counterexample condition (AND-conjunction).
    // If ANY constraint has lb(C_i @ y) > rhs_i, that constraint can never
    // be satisfied, breaking the AND → no counterexample → safe (unsat).
    torch::Tensor lowerDiff = lb - thresholds;
    bool anyDisproved = (lowerDiff > 0).any().item<bool>();

    if (anyDisproved) {
        detail = "lower bound > threshold for at least one constraint (counterexample conjunction broken)";
        return PropertyStatus::Unsat;
    }

    // If ALL constraints have ub(C_i @ y) <= rhs_i, every constraint is always
    // satisfiable → counterexample always exists → definitely unsafe (sat).
    torch::Tensor upperDiff = ub - thresholds;
    bool allSatisfiable = (upperDiff <= 0).all().item<bool>();

    if (allSatisfiable) {
        detail = "all upper bounds <= threshold (counterexample always satisfiable)";
        return PropertyStatus::Sat;
    }

    detail = "bounds do not prove safety or find counterexample";
    return PropertyStatus::Unknown;
}

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <onnx_file> <vnnlib_file> [options]" << std::endl;
    std::cout << "   or: " << programName << " --input <onnx_file> --vnnlib <vnnlib_file> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Positional arguments:" << std::endl;
    std::cout << "  onnx_file          Path to ONNX model file" << std::endl;
    std::cout << "  vnnlib_file        Path to VNN-LIB property file" << std::endl;
    std::cout << std::endl;
    std::cout << "Flag-based arguments:" << std::endl;
    std::cout << "  --input <path>                  Input ONNX model path" << std::endl;
    std::cout << "  --vnnlib <path>                 VNN-LIB property path (alias for --property)" << std::endl;
    std::cout << "  --property <path>               VNN-LIB property path" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --method <crown|alpha-crown>    Analysis method (default: crown)" << std::endl;
    std::cout << "  --iterations <n>                Alpha-CROWN iterations (default: 20)" << std::endl;
    std::cout << "  --lr <float>                    Learning rate (default: 0.5)" << std::endl;
    std::cout << "  --lr-decay <float>              LR decay factor (default: 0.98)" << std::endl;
    std::cout << "  --timeout <seconds>             Global timeout (default: 0 = none)" << std::endl;
    std::cout << "  --seed <n>                      Random seed (default: 1)" << std::endl;
    std::cout << "  --verbose / --quiet             Verbosity control" << std::endl;
    std::cout << "  --optimize-lower / --no-optimize-lower" << std::endl;
    std::cout << "  --optimize-upper / --no-optimize-upper" << std::endl;
    std::cout << "  --enable-first-linear-ibp / --disable-first-linear-ibp" << std::endl;
    std::cout << "  --standard-crown / --crown-ibp  CROWN mode selection" << std::endl;
    std::cout << "  --help, -h                      Print this help message" << std::endl;
}

int lunaMain(int argc, char* argv[]) {
    // Check for help flag
    if (argc > 1) {
        std::string firstArg = argv[1];
        if (firstArg == "--help" || firstArg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Parse arguments - support both positional and flag-based
    std::string onnxFilePath;
    std::string vnnlibFilePath;
    bool useFlags = false;

    // Check if using flag-based arguments (--input, --property, --vnnlib)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" || arg == "--property" || arg == "--vnnlib") {
            useFlags = true;
            break;
        }
    }

    if (useFlags) {
        // Parse flag-based arguments first
        LunaConfiguration::parseArgs(argc, argv);
        
        // Get file paths from configuration
        if (LunaConfiguration::INPUT_FILE_PATH.length() == 0) {
            std::cerr << "ERROR: --input flag is required" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        if (LunaConfiguration::PROPERTY_FILE_PATH.length() == 0) {
            std::cerr << "ERROR: --property or --vnnlib flag is required" << std::endl;
            printUsage(argv[0]);
            return 1;
        }
        
        onnxFilePath = LunaConfiguration::INPUT_FILE_PATH.ascii();
        vnnlibFilePath = LunaConfiguration::PROPERTY_FILE_PATH.ascii();
    } else {
        // Parse positional arguments
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        onnxFilePath = argv[1];
        vnnlibFilePath = argv[2];

        // Prepare arguments for LunaConfiguration::parseArgs
        // We need to skip the first two positional arguments
        std::vector<char*> configArgs;
        configArgs.push_back(argv[0]);  // program name
        for (int i = 3; i < argc; ++i) {
            configArgs.push_back(argv[i]);
        }

        // Parse configuration arguments
        LunaConfiguration::parseArgs(static_cast<int>(configArgs.size()), configArgs.data());
    }

    try {
        if (LunaConfiguration::VERBOSE) {
            std::cout << "Set threads to 1 for deterministic results" << std::endl;
        }

        if (LunaConfiguration::VERBOSE) {
            if (torch::cuda::is_available()) {
                std::cout << "CUDA available with "
                          << torch::cuda::device_count() << " devices" << std::endl;
            }
            std::cout << "Using device: "
                      << LunaConfiguration::getDevice().str() << std::endl;
        }

        std::cout << "Model:    " << onnxFilePath << std::endl;
        std::cout << "Property: " << vnnlibFilePath << std::endl;

        // Step 1: Create TorchModel from ONNX and VNN-LIB files
        std::shared_ptr<NLR::TorchModel> torchModel = std::make_shared<NLR::TorchModel>(
            String(onnxFilePath.c_str()),
            String(vnnlibFilePath.c_str())
        );

        if (LunaConfiguration::VERBOSE) {
            std::cout << "TorchModel created successfully" << std::endl;
            std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
            std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
            std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;
        }

        // Step 2: Verify input bounds were loaded from VNN-LIB file
        if (!torchModel->hasInputBounds()) {
            std::cerr << "ERROR: Input bounds not loaded from VNN-LIB file!" << std::endl;
            return 1;
        }

        BoundedTensor<torch::Tensor> inputBounds = torchModel->getInputBounds();

        if (LunaConfiguration::VERBOSE) {
            std::cout << "\nInput bounds loaded from VNN-LIB file" << std::endl;
            torch::Tensor lowerBounds = inputBounds.lower();
            torch::Tensor upperBounds = inputBounds.upper();
            std::cout << "  Input bounds: " << lowerBounds.size(0) << " variables bounded in ["
                      << lowerBounds.min().item<double>() << ", "
                      << upperBounds.max().item<double>() << "]" << std::endl;
        }

        // Step 3: Check if specification matrix was loaded from VNN-LIB file
        // Note: Do NOT pass the spec matrix to compute_bounds() — it's already stored
        // in the model via setSpecificationFromConstraints() (which preserves thresholds).
        // Passing it again would call setSpecificationMatrix() which clears thresholds
        // needed for early stopping (isSpecVerified).
        if (torchModel->hasSpecificationMatrix()) {
            if (LunaConfiguration::VERBOSE) {
                torch::Tensor C = torchModel->getSpecificationMatrix();
                std::cout << "\nUsing specification matrix from VNN-LIB file with "
                          << C.size(0) << " constraints" << std::endl;
            }
        } else {
            if (LunaConfiguration::VERBOSE) {
                std::cout << "\nNo output constraints found in VNN-LIB file, computing raw output bounds" << std::endl;
            }
        }

        // Step 4: Run analysis based on configured method
        BoundedTensor<torch::Tensor> result;

        // Print config summary
        {
            std::string method = (LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN)
                ? "alpha-crown" : "crown";
            std::cout << "Method:   " << method;
            if (method == "alpha-crown") {
                std::cout << " (iters=" << LunaConfiguration::ALPHA_ITERATIONS
                          << ", lr=" << LunaConfiguration::ALPHA_LR
                          << ", lr_decay=" << LunaConfiguration::ALPHA_LR_DECAY
                          << ", patience=" << LunaConfiguration::EARLY_STOP_PATIENCE << ")";
            }
            std::cout << std::endl;
        }

        if (LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN) {
            result = torchModel->compute_bounds(
                inputBounds,
                nullptr,  // Spec already set via setSpecificationFromConstraints
                LunaConfiguration::AnalysisMethod::AlphaCROWN,
                LunaConfiguration::COMPUTE_LOWER,
                LunaConfiguration::COMPUTE_UPPER
            );
        } else {
            result = torchModel->compute_bounds(
                inputBounds,
                nullptr,  // Spec already set via setSpecificationFromConstraints
                LunaConfiguration::AnalysisMethod::CROWN,
                LunaConfiguration::COMPUTE_LOWER,
                LunaConfiguration::COMPUTE_UPPER
            );
        }

        // Step 5: Output the bounds
        if (result.lower().defined() && result.upper().defined()) {
            printBounds(result.lower(), result.upper());
        } else {
            std::cerr << "ERROR: Bounds are undefined" << std::endl;
            return 1;
        }

        // Step 6: Verify property if constraints exist in VNN-LIB
        PropertyStatus status = PropertyStatus::Unknown;
        std::string statusDetail;
        try {
            NLR::OutputConstraintSet outputConstraints =
                VnnLibInputParser::parseOutputConstraints(
                    String(vnnlibFilePath.c_str()),
                    torchModel->getOutputSize());
            status = evaluatePropertyStatus(outputConstraints,
                                            result.lower(),
                                            result.upper(),
                                            statusDetail);
        } catch (const std::exception& e) {
            statusDetail = std::string("failed to parse output constraints: ") + e.what();
            status = PropertyStatus::Unknown;
        }

        std::string statusLabel;
        switch (status) {
            case PropertyStatus::Unsat:
                statusLabel = "unsat";
                break;
            case PropertyStatus::Sat:
                statusLabel = "sat";
                break;
            case PropertyStatus::Unknown:
            default:
                statusLabel = "unknown";
                break;
        }

        std::cout << "\nResult: " << statusLabel << std::endl;

        return 0;

    } catch (const LunaError& e) {
        std::cerr << "Error: " << e.getErrorClass() << " (code " << e.getCode() << ")" << std::endl;
        if (e.getUserMessage() && strlen(e.getUserMessage()) > 0) {
            std::cerr << "Message: " << e.getUserMessage() << std::endl;
        }
        return 1;
    } catch (const Error& e) {
        std::cerr << "Error: " << e.getErrorClass() << " (code " << e.getCode() << ")" << std::endl;
        if (e.getUserMessage() && strlen(e.getUserMessage()) > 0) {
            std::cerr << "Message: " << e.getUserMessage() << std::endl;
        }
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Error: Unknown exception occurred" << std::endl;
        return 1;
    }
}

