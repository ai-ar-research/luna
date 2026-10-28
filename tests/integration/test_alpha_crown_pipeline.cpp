#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AlphaCROWNPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::manual_seed( 42 );
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
        LunaConfiguration::ALPHA_ITERATIONS = 5;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = false;
    }
};

TEST_F( AlphaCROWNPipelineTest, FullPipeline )
{
    auto model = ModelBuilder::createMLP( 4, { 8, 8 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 4 } ), 0.1 );

    model->setInputBounds( inputBounds );

    // Run AlphaCROWN
    auto outputBounds = model->runAlphaCROWN( true, false );

    // Verify bounds computed
    EXPECT_GT( outputBounds.lower().numel(), 0 );
    EXPECT_GT( outputBounds.upper().numel(), 0 );

    // Verify soundness
    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, outputBounds, *model ) );
}

TEST_F( AlphaCROWNPipelineTest, AlphaCROWNTighterThanCROWN )
{
    auto model = ModelBuilder::createMLP( 5, { 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.15 );
    model->setInputBounds( inputBounds );

    // Get CROWN bounds
    auto crownBounds = model->runCROWN();

    // Get AlphaCROWN bounds
    auto alphaBounds = model->runAlphaCROWN( true, false );

    // AlphaCROWN should be at least as tight
    EXPECT_TRUE( SoundnessChecker::alphaCrownTighterThanCrown( crownBounds, alphaBounds ) );
}
