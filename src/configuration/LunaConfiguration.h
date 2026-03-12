#ifndef __LunaConfiguration_h__
#define __LunaConfiguration_h__

#include "MString.h"
#include "ConfigurationError.h"
#include <torch/torch.h>

class LunaConfiguration
{
public:
    // Enums
    enum class AnalysisMethod {
        CROWN,
        AlphaCROWN
    };

    enum class BoundSide {
        Lower,
        Upper
    };

    // Static methods
    static void print();
    static void resetToDefaults();
    static void parseArgs(int argc, char** argv);

    // Analysis settings
    static AnalysisMethod ANALYSIS_METHOD;
    static bool COMPUTE_LOWER;
    static bool COMPUTE_UPPER;
    static bool VERBOSE;

    // Alpha-CROWN settings
    static unsigned ALPHA_ITERATIONS;
    static float ALPHA_LR;
    static float ALPHA_LR_DECAY;
    static bool KEEP_BEST;
    static bool USE_SHARED_ALPHA;
    static unsigned EARLY_STOP_PATIENCE;
    static bool FIX_INTERM_BOUNDS;
    static bool STABILIZE_INTERMEDIATE_BOUNDS;
    static String OPTIMIZER;
    static float START_SAVE_BEST;
    static BoundSide BOUND_SIDE;
    static bool OPTIMIZE_LOWER;
    static bool OPTIMIZE_UPPER;
    static bool STOP_CROWN_ON_VERIFIED;
    static bool STOP_ALPHA_ON_VERIFIED;

    // CROWN settings
    static bool ENABLE_FIRST_LINEAR_IBP;
    static bool USE_STANDARD_CROWN;

    // Runtime options
    static int VERBOSITY;
    static int TIMEOUT;
    static int SEED;
    static int NUM_BLAS_THREADS;
    static String INPUT_FILE_PATH;
    static String PROPERTY_FILE_PATH;

    // Device options
    static bool USE_CUDA;
    static int CUDA_DEVICE_ID;
    static torch::Device DEVICE;

    static torch::Device getDevice();
    static void updateDeviceFromFlags();

    // Utility constants (replacing GlobalConfiguration constants)
    static const double DEFAULT_EPSILON_FOR_COMPARISONS;
    static const unsigned DEFAULT_DOUBLE_TO_STRING_PRECISION;
    static bool NETWORK_LEVEL_REASONER_LOGGING;
    static const double SIGMOID_CUTOFF_CONSTANT;

    // Helper methods for string conversion
    static String analysisMethodToString(AnalysisMethod method);
    static AnalysisMethod stringToAnalysisMethod(const String& str);
    static String boundSideToString(BoundSide side);
    static BoundSide stringToBoundSide(const String& str);
};

#endif // __LunaConfiguration_h__
