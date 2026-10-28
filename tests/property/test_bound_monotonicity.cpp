#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class BoundMonotonicityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::manual_seed( 42 );
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( BoundMonotonicityTest, TighterInputsProduceTighterOutputs )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );

    // Wide bounds
    auto wideBounds = BoundGenerator::wideBounds( { 1, 3 }, 0.0, 1.0 );
    model->setInputBounds( wideBounds );
    auto wideOutput = model->runCROWN();

    // Narrow bounds (subset of wide)
    auto narrowBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.5 );
    model->setInputBounds( narrowBounds );
    auto narrowOutput = model->runCROWN();

    // Narrow output bounds should be contained in wide output bounds
    // (i.e., narrow lower >= wide lower, narrow upper <= wide upper)
    torch::Tensor narrowWidth = narrowOutput.upper() - narrowOutput.lower();
    torch::Tensor wideWidth = wideOutput.upper() - wideOutput.lower();

    // Width should generally be smaller for tighter inputs
    // (allowing some numerical tolerance)
    EXPECT_TRUE( torch::all( narrowWidth <= wideWidth + 1e-3 ).item<bool>() );
}
