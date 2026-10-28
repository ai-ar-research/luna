#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class CROWNPipelineTest : public ::testing::Test
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
    }
};

TEST_F( CROWNPipelineTest, SimpleMLP )
{
    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 5 } ), 0.1 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    // Verify bounds computed
    EXPECT_GT( outputBounds.lower().numel(), 0 );
    EXPECT_GT( outputBounds.upper().numel(), 0 );

    // Verify soundness
    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, outputBounds, *model ) );
}

TEST_F( CROWNPipelineTest, DeeperMLP )
{
    auto model = ModelBuilder::createMLP( 10, { 20, 20, 20 }, 5, true, false );

    auto inputBounds = BoundGenerator::randomBounds( { 1, 10 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    // Verify soundness by sampling
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, outputBounds, 100 ) );
}
