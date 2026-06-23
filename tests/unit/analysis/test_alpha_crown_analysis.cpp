#include <gtest/gtest.h>
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AlphaCROWNAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        static bool threadsConfigured = false;
        if (!threadsConfigured) {
            at::set_num_threads(1);
            at::set_num_interop_threads(1);
            threadsConfigured = true;
        }
        
        // Configure AlphaCROWN
        LunaConfiguration::ALPHA_ITERATIONS = 5;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = false;
    }
};

TEST_F(AlphaCROWNAnalysisTest, BasicOptimization) {
    auto model = ModelBuilder::createMLP(3, {5, 5}, 2, true, false);
    
    auto inputBounds = BoundGenerator::epsilonBall(
        torch::zeros({1, 3}), 0.1);
    model->setInputBounds(inputBounds);
    
    NLR::AlphaCROWNAnalysis analysis(model.get());
    analysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    
    // Test lower bound optimization
    torch::Tensor lowerBounds = analysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);
    
    EXPECT_GT(lowerBounds.numel(), 0);
}

TEST_F(AlphaCROWNAnalysisTest, AlphaCROWNTighterThanCROWN) {
    auto model = ModelBuilder::createMLP(4, {6}, 3, true, false);

    auto inputBounds = BoundGenerator::randomBounds({1, 4}, 0.0, 1.0, 0.15);
    model->setInputBounds(inputBounds);

    // Get CROWN bounds
    NLR::CROWNAnalysis crownAnalysis(model.get());
    crownAnalysis.setInputBounds(inputBounds);
    crownAnalysis.run();
    auto crownBounds = crownAnalysis.getOutputBounds();

    // Get AlphaCROWN bounds
    NLR::AlphaCROWNAnalysis alphaAnalysis(model.get());
    alphaAnalysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    torch::Tensor alphaLower = alphaAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);
    torch::Tensor alphaUpper = alphaAnalysis.getCROWNAnalysis()->getOutputBounds().upper();

    BoundedTensor<torch::Tensor> alphaBounds(alphaLower, alphaUpper);

    // AlphaCROWN should be at least as tight as CROWN
    EXPECT_TRUE(SoundnessChecker::alphaCrownTighterThanCrown(crownBounds, alphaBounds));
}

TEST_F(AlphaCROWNAnalysisTest, DeepNetworkAlphaCROWNTighterThanCROWN) {
    auto model = ModelBuilder::createMLP(5, {10, 10, 10}, 3, true, false);

    auto inputBounds = BoundGenerator::epsilonBall(
        torch::randn({1, 5}) * 0.5, 0.1);
    model->setInputBounds(inputBounds);

    LunaConfiguration::ALPHA_ITERATIONS = 10;

    NLR::CROWNAnalysis crownAnalysis(model.get());
    crownAnalysis.setInputBounds(inputBounds);
    crownAnalysis.run();
    auto crownBounds = crownAnalysis.getOutputBounds();

    NLR::AlphaCROWNAnalysis alphaAnalysis(model.get());
    alphaAnalysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    torch::Tensor alphaLower = alphaAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);
    torch::Tensor alphaUpper = alphaAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Upper);

    BoundedTensor<torch::Tensor> alphaBounds(alphaLower, alphaUpper);

    EXPECT_TRUE(SoundnessChecker::alphaCrownTighterThanCrown(crownBounds, alphaBounds));
}

TEST_F(AlphaCROWNAnalysisTest, RecomputeIntermediateBoundsProducesTighterBounds) {
    // Recomputing intermediate bounds each iteration should produce tighter
    // (or equal) bounds vs freezing them after the initial CROWN pass.
    torch::manual_seed(123);
    auto model = ModelBuilder::createMLP(5, {15, 15, 15}, 3, true, false);

    auto inputBounds = BoundGenerator::epsilonBall(
        torch::randn({1, 5}) * 0.3, 0.15);
    model->setInputBounds(inputBounds);

    LunaConfiguration::ALPHA_ITERATIONS = 15;
    LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS = false;

    // Run with frozen intermediate bounds
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = false;
    NLR::AlphaCROWNAnalysis frozenAnalysis(model.get());
    frozenAnalysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    torch::Tensor frozenLower = frozenAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);

    // Run with recomputed intermediate bounds
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
    NLR::AlphaCROWNAnalysis recomputeAnalysis(model.get());
    recomputeAnalysis.getCROWNAnalysis()->setInputBounds(inputBounds);
    torch::Tensor recomputeLower = recomputeAnalysis.computeOptimizedBounds(
        LunaConfiguration::BoundSide::Lower);

    // Recomputed should be at least as tight (higher lower bounds)
    auto recompFlat = recomputeLower.flatten();
    auto frozenFlat = frozenLower.flatten();
    float tolerance = 1e-4f;

    for (int64_t i = 0; i < recompFlat.numel(); ++i) {
        EXPECT_GE(recompFlat[i].item<float>(), frozenFlat[i].item<float>() - tolerance)
            << "Spec " << i << ": recomputed lower bound should be >= frozen lower bound";
    }

    LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS = true;
    LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS = true;
}

TEST(AlphaCROWNConfigDefaults, RecomputeIntermediateBoundsDefaultsToTrue) {
    // Intermediate bounds must be recomputed by default so alpha optimization
    // can tighten them across iterations. Freezing them (false) degrades bounds.
    EXPECT_TRUE(LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS);
}
