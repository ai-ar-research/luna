#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedFlattenNode.h"
#include "src/common/BoundedTensor.h"
#include "src/input_parsers/Operations.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class FlattenNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        Operations::FlattenWrapper flattenWrapper( 1 );
        _flattenNode = std::make_shared<BoundedFlattenNode>( flattenWrapper );
        _flattenNode->setNodeIndex( 0 );
        _flattenNode->setNodeName( "test_flatten" );
    }

    std::shared_ptr<BoundedFlattenNode> _flattenNode;
};

TEST_F( FlattenNodeTest, ForwardPass2D )
{
    torch::Tensor input = torch::randn( { 2, 3, 4, 5 } ); // (batch, channels, height, width)
    torch::Tensor output = _flattenNode->forward( input );

    // Should flatten to (batch, channels*height*width)
    EXPECT_EQ( output.size( 0 ), 2 );
    EXPECT_EQ( output.size( 1 ), 3 * 4 * 5 );
}

TEST_F( FlattenNodeTest, IBPPreservesValues )
{
    torch::Tensor lb = torch::zeros( { 1, 2, 3, 4 } );
    torch::Tensor ub = torch::ones( { 1, 2, 3, 4 } );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _flattenNode->computeIntervalBoundPropagation( inputBounds );

    // Flattened bounds should have same values, just reshaped
    EXPECT_EQ( result.lower().numel(), lb.numel() );
    EXPECT_EQ( result.upper().numel(), ub.numel() );
}
