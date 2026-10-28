#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class RelaxationSoundnessTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::manual_seed( 42 );
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( RelaxationSoundnessTest, ReLURelaxationSound )
{
    // Test that ReLU relaxations produce valid overbounds
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 3 }, -1.0, 1.0, 0.5 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    // Verify soundness - relaxations should overbound true outputs
    EXPECT_TRUE(
        SoundnessChecker::verifySoundnessBySampling( *model, inputBounds, outputBounds, 1000 ) );
}
