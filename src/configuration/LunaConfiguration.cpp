#include "LunaConfiguration.h"
#include "MString.h"
#include "ConfigurationError.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <torch/torch.h>
#include <string>

// Initialize static members with default values

// Analysis settings
LunaConfiguration::AnalysisMethod LunaConfiguration::ANALYSIS_METHOD = 
    LunaConfiguration::AnalysisMethod::CROWN;
bool LunaConfiguration::COMPUTE_LOWER = true;
bool LunaConfiguration::COMPUTE_UPPER = true;
bool LunaConfiguration::VERBOSE = false;

// Alpha-CROWN settings
unsigned LunaConfiguration::ALPHA_ITERATIONS = 20;
float LunaConfiguration::ALPHA_LR = 0.5f;
float LunaConfiguration::ALPHA_LR_DECAY = 0.98f;
bool LunaConfiguration::KEEP_BEST = true;
bool LunaConfiguration::USE_SHARED_ALPHA = false;
unsigned LunaConfiguration::EARLY_STOP_PATIENCE = 10;
bool LunaConfiguration::FIX_INTERM_BOUNDS = true;
bool LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS = true;
String LunaConfiguration::OPTIMIZER = "adam";
float LunaConfiguration::START_SAVE_BEST = 0.5f;
LunaConfiguration::BoundSide LunaConfiguration::BOUND_SIDE = 
    LunaConfiguration::BoundSide::Lower;
bool LunaConfiguration::OPTIMIZE_LOWER = true;
bool LunaConfiguration::OPTIMIZE_UPPER = false;

// CROWN settings
bool LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = true;
bool LunaConfiguration::USE_STANDARD_CROWN = true;

// Runtime options
int LunaConfiguration::VERBOSITY = 0;
int LunaConfiguration::TIMEOUT = 0;
int LunaConfiguration::SEED = 1;
int LunaConfiguration::NUM_BLAS_THREADS = 1;
String LunaConfiguration::INPUT_FILE_PATH = "";
String LunaConfiguration::PROPERTY_FILE_PATH = "";

// Device options
bool LunaConfiguration::USE_CUDA = torch::cuda::is_available();
int LunaConfiguration::CUDA_DEVICE_ID = 0;
torch::Device LunaConfiguration::DEVICE =
    LunaConfiguration::USE_CUDA ? torch::Device(torch::kCUDA, CUDA_DEVICE_ID)
                                 : torch::Device(torch::kCPU);

// Utility constants (replacing GlobalConfiguration constants)
const double LunaConfiguration::DEFAULT_EPSILON_FOR_COMPARISONS = 0.0000000001;
const unsigned LunaConfiguration::DEFAULT_DOUBLE_TO_STRING_PRECISION = 10;
bool LunaConfiguration::NETWORK_LEVEL_REASONER_LOGGING = false;
const double LunaConfiguration::SIGMOID_CUTOFF_CONSTANT = 20.0;

// Helper methods for string conversion
String LunaConfiguration::analysisMethodToString(AnalysisMethod method)
{
    switch (method) {
        case AnalysisMethod::CROWN:
            return "crown";
        case AnalysisMethod::AlphaCROWN:
            return "alpha-crown";
        default:
            return "unknown";
    }
}

LunaConfiguration::AnalysisMethod LunaConfiguration::stringToAnalysisMethod(const String& str)
{
    if (str == "crown" || str == "CROWN") {
        return AnalysisMethod::CROWN;
    } else if (str == "alpha-crown" || str == "alpha_crown" || str == "AlphaCROWN") {
        return AnalysisMethod::AlphaCROWN;
    } else {
        return AnalysisMethod::CROWN; // Default
    }
}

String LunaConfiguration::boundSideToString(BoundSide side)
{
    switch (side) {
        case BoundSide::Lower:
            return "lower";
        case BoundSide::Upper:
            return "upper";
        default:
            return "unknown";
    }
}

LunaConfiguration::BoundSide LunaConfiguration::stringToBoundSide(const String& str)
{
    if (str == "lower" || str == "Lower" || str == "LOWER") {
        return BoundSide::Lower;
    } else if (str == "upper" || str == "Upper" || str == "UPPER") {
        return BoundSide::Upper;
    } else {
        return BoundSide::Lower; // Default
    }
}

void LunaConfiguration::resetToDefaults()
{
    // Analysis settings
    ANALYSIS_METHOD = AnalysisMethod::CROWN;
    COMPUTE_LOWER = true;
    COMPUTE_UPPER = true;
    VERBOSE = false;

    // Alpha-CROWN settings
    ALPHA_ITERATIONS = 20;
    ALPHA_LR = 0.5f;
    ALPHA_LR_DECAY = 0.98f;
    KEEP_BEST = true;
    USE_SHARED_ALPHA = false;
    EARLY_STOP_PATIENCE = 10;
    FIX_INTERM_BOUNDS = true;
    STABILIZE_INTERMEDIATE_BOUNDS = true;
    OPTIMIZER = "adam";
    START_SAVE_BEST = 0.5f;
    BOUND_SIDE = BoundSide::Lower;
    OPTIMIZE_LOWER = true;
    OPTIMIZE_UPPER = false;

    // CROWN settings
    ENABLE_FIRST_LINEAR_IBP = true;
    USE_STANDARD_CROWN = true;

    // Runtime options
    VERBOSITY = 2;
    TIMEOUT = 0;
    SEED = 1;
    NUM_BLAS_THREADS = 1;
    INPUT_FILE_PATH = "";
    PROPERTY_FILE_PATH = "";

    // Device options
    USE_CUDA = torch::cuda::is_available();
    CUDA_DEVICE_ID = 0;
    updateDeviceFromFlags();
    
    // Utility constants are const, so they don't need resetting
    // NETWORK_LEVEL_REASONER_LOGGING can be reset if needed
    NETWORK_LEVEL_REASONER_LOGGING = false;
}

void LunaConfiguration::print()
{
    printf("*** LUNA Configuration ***\n");

    // Analysis settings
    printf("Analysis Settings:\n");
    printf("  ANALYSIS_METHOD: %s\n", analysisMethodToString(ANALYSIS_METHOD).ascii());
    printf("  COMPUTE_LOWER: %s\n", COMPUTE_LOWER ? "true" : "false");
    printf("  COMPUTE_UPPER: %s\n", COMPUTE_UPPER ? "true" : "false");
    printf("  VERBOSE: %s\n", VERBOSE ? "true" : "false");
    
    // Alpha-CROWN settings
    printf("\nAlpha-CROWN Settings:\n");
    printf("  ALPHA_ITERATIONS: %u\n", ALPHA_ITERATIONS);
    printf("  ALPHA_LR: %.3f\n", ALPHA_LR);
    printf("  ALPHA_LR_DECAY: %.3f\n", ALPHA_LR_DECAY);
    printf("  KEEP_BEST: %s\n", KEEP_BEST ? "true" : "false");
    printf("  USE_SHARED_ALPHA: %s\n", USE_SHARED_ALPHA ? "true" : "false");
    printf("  EARLY_STOP_PATIENCE: %u\n", EARLY_STOP_PATIENCE);
    printf("  FIX_INTERM_BOUNDS: %s\n", FIX_INTERM_BOUNDS ? "true" : "false");
    printf("  STABILIZE_INTERMEDIATE_BOUNDS: %s\n", STABILIZE_INTERMEDIATE_BOUNDS ? "true" : "false");
    printf("  OPTIMIZER: %s\n", OPTIMIZER.ascii());
    printf("  START_SAVE_BEST: %.3f\n", START_SAVE_BEST);
    printf("  BOUND_SIDE: %s\n", boundSideToString(BOUND_SIDE).ascii());
    printf("  OPTIMIZE_LOWER: %s\n", OPTIMIZE_LOWER ? "true" : "false");
    printf("  OPTIMIZE_UPPER: %s\n", OPTIMIZE_UPPER ? "true" : "false");
    
    // CROWN settings
    printf("\nCROWN Settings:\n");
    printf("  ENABLE_FIRST_LINEAR_IBP: %s\n", ENABLE_FIRST_LINEAR_IBP ? "true" : "false");
    printf("  USE_STANDARD_CROWN: %s\n", USE_STANDARD_CROWN ? "true" : "false");
    
    // Runtime options
    printf("\nRuntime Options:\n");
    printf("  VERBOSITY: %d\n", VERBOSITY);
    printf("  TIMEOUT: %d\n", TIMEOUT);
    printf("  SEED: %d\n", SEED);
    printf("  NUM_BLAS_THREADS: %d\n", NUM_BLAS_THREADS);
    printf("  INPUT_FILE_PATH: %s\n", INPUT_FILE_PATH.ascii());
    printf("  PROPERTY_FILE_PATH: %s\n", PROPERTY_FILE_PATH.ascii());

    // Device options
    printf("\nDevice Options:\n");
    printf("  USE_CUDA: %s\n", USE_CUDA ? "true" : "false");
    printf("  CUDA_DEVICE_ID: %d\n", CUDA_DEVICE_ID);
    printf("  DEVICE: %s\n", DEVICE.str().c_str());
    
}

torch::Device LunaConfiguration::getDevice()
{
    return DEVICE;
}

void LunaConfiguration::updateDeviceFromFlags()
{
    if (USE_CUDA && !torch::cuda::is_available()) {
        USE_CUDA = false;
    }
    if (USE_CUDA) {
        DEVICE = torch::Device(torch::kCUDA, CUDA_DEVICE_ID);
    } else {
        DEVICE = torch::Device(torch::kCPU);
    }
}

void LunaConfiguration::parseArgs(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i) {
        String arg(argv[i]);
        
        // Method selection
        if (arg == "--method" && i + 1 < argc) {
            ANALYSIS_METHOD = stringToAnalysisMethod(String(argv[++i]));
        }
        // Alpha-CROWN iterations
        else if (arg == "--iterations" && i + 1 < argc) {
            ALPHA_ITERATIONS = static_cast<unsigned>(atoi(argv[++i]));
        }
        // Learning rate
        else if (arg == "--lr" && i + 1 < argc) {
            ALPHA_LR = static_cast<float>(atof(argv[++i]));
        }
        // Learning rate decay
        else if (arg == "--lr-decay" && i + 1 < argc) {
            ALPHA_LR_DECAY = static_cast<float>(atof(argv[++i]));
        }
        // Timeout
        else if (arg == "--timeout" && i + 1 < argc) {
            TIMEOUT = atoi(argv[++i]);
        }
        // Seed
        else if (arg == "--seed" && i + 1 < argc) {
            SEED = atoi(argv[++i]);
        }
        // Verbosity level
        else if (arg == "--verbosity" && i + 1 < argc) {
            VERBOSITY = atoi(argv[++i]);
        }
        // Verbose
        else if (arg == "--verbose") {
            VERBOSE = true;
        }
        // Quiet
        else if (arg == "--quiet") {
            VERBOSE = false;
        }
        // Input file
        else if (arg == "--input" && i + 1 < argc) {
            INPUT_FILE_PATH = String(argv[++i]);
        }
        // Property file
        else if (arg == "--property" && i + 1 < argc) {
            PROPERTY_FILE_PATH = String(argv[++i]);
        }
        // VNN-LIB file (alias for --property)
        else if (arg == "--vnnlib" && i + 1 < argc) {
            PROPERTY_FILE_PATH = String(argv[++i]);
        }
        // Device selection (explicit)
        else if (arg == "--device" && i + 1 < argc) {
            String deviceStr(argv[++i]);
            if (deviceStr == "cpu" || deviceStr == "CPU") {
                USE_CUDA = false;
            } else if (deviceStr.startsWith("cuda") || deviceStr.startsWith("CUDA")) {
                USE_CUDA = true;
                size_t colonIndex = deviceStr.find(":");
                if (colonIndex != std::string::npos) {
                    unsigned start = static_cast<unsigned>(colonIndex + 1);
                    unsigned count = deviceStr.length() - start;
                    String idStr = deviceStr.substring(start, count);
                    CUDA_DEVICE_ID = atoi(idStr.ascii());
                }
            }
            updateDeviceFromFlags();
        }
        // CUDA on
        else if (arg == "--cuda") {
            USE_CUDA = true;
            updateDeviceFromFlags();
        }
        // CPU only
        else if (arg == "--cpu") {
            USE_CUDA = false;
            updateDeviceFromFlags();
        }
        // CUDA device id
        else if (arg == "--cuda-device" && i + 1 < argc) {
            CUDA_DEVICE_ID = atoi(argv[++i]);
            updateDeviceFromFlags();
        }
        // Optimize lower
        else if (arg == "--optimize-lower") {
            OPTIMIZE_LOWER = true;
        }
        // No optimize lower
        else if (arg == "--no-optimize-lower") {
            OPTIMIZE_LOWER = false;
        }
        // Optimize upper
        else if (arg == "--optimize-upper") {
            OPTIMIZE_UPPER = true;
        }
        // No optimize upper
        else if (arg == "--no-optimize-upper") {
            OPTIMIZE_UPPER = false;
        }
        // Enable first linear IBP
        else if (arg == "--enable-first-linear-ibp") {
            ENABLE_FIRST_LINEAR_IBP = true;
        }
        // Disable first linear IBP
        else if (arg == "--disable-first-linear-ibp") {
            ENABLE_FIRST_LINEAR_IBP = false;
        }
        // Stabilize intermediate bounds (STE)
        else if (arg == "--stabilize") {
            STABILIZE_INTERMEDIATE_BOUNDS = true;
        }
        else if (arg == "--no-stabilize") {
            STABILIZE_INTERMEDIATE_BOUNDS = false;
        }
        // Standard CROWN
        else if (arg == "--standard-crown") {
            USE_STANDARD_CROWN = true;
        }
        // CROWN-IBP
        else if (arg == "--crown-ibp") {
            USE_STANDARD_CROWN = false;
        }
        // Help
        else if (arg == "--help" || arg == "-h") {
            printf("LUNA Configuration Options:\n");
            printf("  --method <crown|alpha-crown>    Analysis method (default: crown)\n");
            printf("  --iterations <n>                Alpha-CROWN iterations (default: 20)\n");
            printf("  --lr <float>                    Learning rate (default: 0.5)\n");
            printf("  --lr-decay <float>              LR decay factor (default: 0.98)\n");
            printf("  --timeout <seconds>             Global timeout (default: 0 = none)\n");
            printf("  --seed <n>                      Random seed (default: 1)\n");
            printf("  --verbosity <n>                 Verbosity level (0=quiet, 1+=verbose, default: 0)\n");
            printf("  --verbose / --quiet             Verbosity control (legacy)\n");
            printf("  --input <path>                  Input ONNX model path\n");
            printf("  --property <path>               VNN-LIB property path\n");
            printf("  --device <cpu|cuda|cuda:N>      Device selection (default: auto)\n");
            printf("  --cuda / --cpu                  Force CUDA or CPU\n");
            printf("  --cuda-device <n>               CUDA device index (default: 0)\n");
            printf("  --optimize-lower / --no-optimize-lower\n");
            printf("  --optimize-upper / --no-optimize-upper\n");
            printf("  --stabilize / --no-stabilize    STE stabilization for intermediate bounds (default: on)\n");
            printf("  --enable-first-linear-ibp / --disable-first-linear-ibp\n");
            printf("  --standard-crown / --crown-ibp  CROWN mode selection\n");
            printf("  --help                          Print this help message\n");
            exit(0);
        }
    }
}