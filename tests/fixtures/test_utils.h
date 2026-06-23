#ifndef __TEST_UTILS_H__
#define __TEST_UTILS_H__

#include "src/engine/TorchModel.h"
#include "src/common/BoundedTensor.h"
#include "tensor_comparators.h"
#include <torch/torch.h>
#include <memory>
#include <random>

namespace test {

/**
 * Generate test input bounds for models.
 */
class BoundGenerator {
public:
    // Generate L∞ epsilon ball around center point
    static BoundedTensor<torch::Tensor> epsilonBall(
        const torch::Tensor& center,
        double epsilon);

    // Generate random bounds with specified shape and range
    static BoundedTensor<torch::Tensor> randomBounds(
        const std::vector<int64_t>& shape,
        double minVal,
        double maxVal,
        double width);

    // Generate wide bounds (useful for testing)
    static BoundedTensor<torch::Tensor> wideBounds(
        const std::vector<int64_t>& shape,
        double center = 0.0,
        double halfWidth = 10.0);

    // Load bounds from VNN-LIB file
    static BoundedTensor<torch::Tensor> fromVNNLib(const std::string& vnnlibPath);
};

/**
 * Mathematical soundness verification utilities.
 */
class SoundnessChecker {
public:
    // Check if a single value is contained within bounds
    static bool boundsContainValue(
        const BoundedTensor<torch::Tensor>& bounds,
        const torch::Tensor& value);

    // Verify soundness by Monte Carlo sampling
    // Returns true if all sampled outputs are within the computed bounds
    static bool verifySoundnessBySampling(
        NLR::TorchModel& model,
        const BoundedTensor<torch::Tensor>& inputBounds,
        const BoundedTensor<torch::Tensor>& outputBounds,
        unsigned numSamples = 1000,
        unsigned seed = 42);

    // Verify center point is contained in bounds
    static bool centerContainedInBounds(
        const BoundedTensor<torch::Tensor>& inputBounds,
        const BoundedTensor<torch::Tensor>& outputBounds,
        NLR::TorchModel& model);

    // Check that CROWN bounds are at least as tight as IBP bounds
    static bool crownTighterThanIBP(
        const BoundedTensor<torch::Tensor>& ibpBounds,
        const BoundedTensor<torch::Tensor>& crownBounds);

    // Check that AlphaCROWN bounds are at least as tight as CROWN bounds
    static bool alphaCrownTighterThanCrown(
        const BoundedTensor<torch::Tensor>& crownBounds,
        const BoundedTensor<torch::Tensor>& alphaCrownBounds);
};

} // namespace test

#define EXPECT_BOUNDS_SOUND(bounds, value) \
    EXPECT_TRUE(test::SoundnessChecker::boundsContainValue(bounds, value))

#define ASSERT_BOUNDS_SOUND(bounds, value) \
    ASSERT_TRUE(test::SoundnessChecker::boundsContainValue(bounds, value))

#endif // __TEST_UTILS_H__
