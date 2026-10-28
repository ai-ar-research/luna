#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>
#include <fstream>
#include <sstream>

using namespace NLR;
using namespace test;

class CROWNVsAutoLiRPATest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }

    // Load reference data from JSON (simplified - would use proper JSON parser)
    bool loadReferenceData( const std::string &jsonPath, BoundedTensor<torch::Tensor> &crownBounds )
    {
        // TODO: parse JSON reference bounds and populate crownBounds.
        return false;
    }
};

TEST_F( CROWNVsAutoLiRPATest, CompareCROWNBounds )
{
    // Test comparing C++ CROWN with Python auto_LiRPA reference
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 3 } ), 0.1 );

    model->setInputBounds( inputBounds );
    auto cppBounds = model->runCROWN();

    // TODO: load reference JSON and compare against cppBounds.
    EXPECT_GT( cppBounds.lower().numel(), 0 );
    EXPECT_GT( cppBounds.upper().numel(), 0 );
}
