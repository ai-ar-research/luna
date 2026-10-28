#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ComprehensiveReferenceTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( ComprehensiveReferenceTest, MultipleModels )
{
    // Test suite across multiple reference models
    // TODO: iterate reference data directory and compare per-model bounds.

    auto model = ModelBuilder::createMLP( 5, { 8 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );
    auto bounds = model->runCROWN();

    EXPECT_GT( bounds.lower().numel(), 0 );
}
