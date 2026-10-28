#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class CROWNSoundnessTest : public ::testing::Test
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

TEST_F( CROWNSoundnessTest, SoundnessSimpleMLP )
{
    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 5 } ), 0.1f );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    // Verify soundness by sampling
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, outputBounds, 1000 ) );
}

TEST_F( CROWNSoundnessTest, CenterContained )
{
    auto model = ModelBuilder::createMLP( 4, { 8 }, 2, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 4 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, outputBounds, *model ) );
}

TEST_F( CROWNSoundnessTest, CROWNTighterThanIBP )
{
    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );
    model->setInputBounds( inputBounds );

    // Get IBP bounds
    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    // Get CROWN bounds
    auto crownBounds = model->runCROWN();

    EXPECT_TRUE( SoundnessChecker::crownTighterThanIBP( ibpBounds, crownBounds ) );
}
