#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AlphaCROWNSoundnessTest : public ::testing::Test
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
    }
};

TEST_F( AlphaCROWNSoundnessTest, AlphaCROWNSoundness )
{
    auto model = ModelBuilder::createMLP( 4, { 8 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 4 } ), 0.1 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runAlphaCROWN( true, false );

    // Verify AlphaCROWN maintains soundness
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, outputBounds, 500 ) );
}

TEST_F( AlphaCROWNSoundnessTest, AlphaCROWNMaintainsTightness )
{
    auto model = ModelBuilder::createMLP( 5, { 10 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );
    auto crownBounds = model->runCROWN();
    auto alphaBounds = model->runAlphaCROWN( true, false );

    // AlphaCROWN should be at least as tight
    EXPECT_TRUE( SoundnessChecker::alphaCrownTighterThanCrown( crownBounds, alphaBounds ) );
}
