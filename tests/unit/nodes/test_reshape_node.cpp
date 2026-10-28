#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedReshapeNode.h"
#include "src/common/BoundedTensor.h"
#include "src/input_parsers/Operations.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ReshapeNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto shapeTensor = torch::tensor( { 24 }, torch::TensorOptions().dtype( torch::kInt64 ) );
        Operations::ReshapeWrapper reshapeWrapper( shapeTensor );
        _reshapeNode = std::make_shared<BoundedReshapeNode>( reshapeWrapper );
        _reshapeNode->setNodeIndex( 0 );
        _reshapeNode->setNodeName( "test_reshape" );
    }

    std::shared_ptr<BoundedReshapeNode> _reshapeNode;
};

TEST_F( ReshapeNodeTest, ForwardPassReshape )
{
    // Input: (1, 2, 3, 4) = 24 elements
    torch::Tensor input = torch::randn( { 1, 2, 3, 4 } );
    torch::Tensor output = _reshapeNode->forward( input );

    // Should reshape to (1, 24)
    EXPECT_EQ( output.size( 0 ), 1 );
    EXPECT_EQ( output.size( 1 ), 24 );
    EXPECT_EQ( output.numel(), input.numel() );
}

TEST_F( ReshapeNodeTest, IBPPreservesValues )
{
    torch::Tensor lb = torch::zeros( { 1, 2, 3, 4 } );
    torch::Tensor ub = torch::ones( { 1, 2, 3, 4 } );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _reshapeNode->computeIntervalBoundPropagation( inputBounds );

    // Reshaped bounds should preserve values
    EXPECT_EQ( result.lower().numel(), lb.numel() );
    EXPECT_EQ( result.upper().numel(), ub.numel() );
}
