#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>
#include <fstream>
#include <json/json.h>

using namespace NLR;
using namespace test;

class ToyModelsRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }

    BoundedTensor<torch::Tensor> loadGoldenBounds( const std::string &jsonPath )
    {
        // TODO: parse JSON golden data and return reference bounds.
        torch::Tensor empty = torch::zeros( { 1 } );
        return BoundedTensor<torch::Tensor>( empty, empty );
    }
};

TEST_F( ToyModelsRegressionTest, SimpleMLPRegression )
{
    // Test that simple models produce consistent outputs
    auto model = ModelBuilder::createMLP( 2, { 3 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 2 } ), 0.1 );

    model->setInputBounds( inputBounds );
    auto outputBounds = model->runCROWN();

    // TODO: compare outputBounds against golden reference bounds.
    EXPECT_GT( outputBounds.lower().numel(), 0 );
    EXPECT_GT( outputBounds.upper().numel(), 0 );
}
