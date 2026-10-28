#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedAddNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AddNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _addNode = std::make_shared<BoundedAddNode>();
        _addNode->setNodeIndex( 0 );
        _addNode->setNodeName( "test_add" );
    }

    std::shared_ptr<BoundedAddNode> _addNode;
};

TEST_F( AddNodeTest, ForwardPassTwoInputs )
{
    torch::Tensor input1 = torch::ones( { 1, 5 } );
    torch::Tensor input2 = torch::ones( { 1, 5 } ) * 2.0;

    std::vector<torch::Tensor> inputs = { input1, input2 };
    torch::Tensor output = _addNode->forward( inputs );

    torch::Tensor expected = torch::ones( { 1, 5 } ) * 3.0;
    EXPECT_TENSORS_CLOSE( output, expected, 1e-5, 1e-4 );
}

TEST_F( AddNodeTest, IBPBasic )
{
    torch::Tensor lb1 = torch::zeros( { 1, 3 } );
    torch::Tensor ub1 = torch::ones( { 1, 3 } );
    torch::Tensor lb2 = torch::zeros( { 1, 3 } );
    torch::Tensor ub2 = torch::ones( { 1, 3 } ) * 2.0;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb1, ub1 ) );
    inputBounds.append( BoundedTensor<torch::Tensor>( lb2, ub2 ) );

    auto result = _addNode->computeIntervalBoundPropagation( inputBounds );

    // Lower bound should be sum of lower bounds
    // Upper bound should be sum of upper bounds
    torch::Tensor expectedLower = lb1 + lb2;
    torch::Tensor expectedUpper = ub1 + ub2;

    EXPECT_TENSORS_CLOSE( result.lower(), expectedLower, 1e-5, 1e-4 );
    EXPECT_TENSORS_CLOSE( result.upper(), expectedUpper, 1e-5, 1e-4 );
}

// A flat coefficient matrix against an unflattened operand holds the same elements, so no broadcast
TEST_F( AddNodeTest, BackwardKeepsFlatFeaturesForAnUnflattenedOperandOfTheSameSize )
{
    torch::Tensor constant = torch::tensor( { 1.0, 2.0, 3.0, 4.0, 5.0 } ).reshape( { 1, 1, 1, 5 } );
    _addNode->setConstantValue( constant );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( torch::zeros( { 1, 1, 1, 5 } ),
                                                      torch::ones( { 1, 1, 1, 5 } ) ) );
    BoundA lastA( torch::ones( { 1, 1, 5 } ) );
    Vector<Pair<BoundA, BoundA>> outputA;
    torch::Tensor lbias;
    torch::Tensor ubias;

    _addNode->boundBackward( lastA, lastA, inputBounds, outputA, lbias, ubias );

    ASSERT_EQ( outputA.size(), 1u );
    EXPECT_EQ( outputA[0].first().asTensor().sizes(), torch::IntArrayRef( { 1, 1, 5 } ) );
    EXPECT_NEAR( lbias.item<double>(), 15.0, 1e-5 );
}

// A scalar operand is broadcast over every feature, so its coefficient is the sum across them
TEST_F( AddNodeTest, BackwardSumsOverFeaturesForABroadcastOperand )
{
    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append(
        BoundedTensor<torch::Tensor>( torch::zeros( { 1, 1 } ), torch::ones( { 1, 1 } ) ) );
    BoundA lastA( torch::ones( { 1, 1, 4 } ) );
    Vector<Pair<BoundA, BoundA>> outputA;
    torch::Tensor lbias;
    torch::Tensor ubias;

    _addNode->boundBackward( lastA, lastA, inputBounds, outputA, lbias, ubias );

    ASSERT_EQ( outputA.size(), 1u );
    const torch::Tensor collapsed = outputA[0].first().asTensor();
    EXPECT_EQ( collapsed.sizes(), torch::IntArrayRef( { 1, 1, 1 } ) );
    EXPECT_NEAR( collapsed.item<double>(), 4.0, 1e-5 );
}
