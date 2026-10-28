#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class IBPSoundnessTest : public ::testing::Test
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

TEST_F( IBPSoundnessTest, IBPSoundnessBySampling )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 3 } ), 0.15 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    // Verify IBP is sound
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, ibpBounds, 500 ) );
}

TEST_F( IBPSoundnessTest, IBPCenterContained )
{
    auto model = ModelBuilder::createMLP( 4, { 6 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 4 }, 0.0, 1.0, 0.25 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, ibpBounds, *model ) );
}
