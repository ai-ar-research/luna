#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedSigmoidNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class SigmoidNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::nn::Sigmoid sigmoid;
        _sigmoidNode = std::make_shared<BoundedSigmoidNode>( sigmoid, "test_sigmoid" );
        _sigmoidNode->setNodeIndex( 0 );
        _sigmoidNode->setInputSize( 5 );
        _sigmoidNode->setOutputSize( 5 );
    }

    std::shared_ptr<BoundedSigmoidNode> _sigmoidNode;
};

TEST_F( SigmoidNodeTest, ForwardPassBasic )
{
    torch::Tensor input = torch::zeros( { 1, 5 } );
    torch::Tensor output = _sigmoidNode->forward( input );

    // sigmoid(0) = 0.5
    torch::Tensor expected = torch::ones( { 1, 5 } ) * 0.5;
    EXPECT_TENSORS_CLOSE( output, expected, 1e-5, 1e-4 );
}

TEST_F( SigmoidNodeTest, IBPRange )
{
    torch::Tensor lb = torch::zeros( { 1, 5 } ) - 5.0;
    torch::Tensor ub = torch::zeros( { 1, 5 } ) + 5.0;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _sigmoidNode->computeIntervalBoundPropagation( inputBounds );

    // Sigmoid output should be in [0, 1]
    EXPECT_TRUE( torch::all( result.lower() >= 0 ).item<bool>() );
    EXPECT_TRUE( torch::all( result.upper() <= 1.0 ).item<bool>() );
}
