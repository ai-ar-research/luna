#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedConvNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ConvNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a 2x3 conv layer with kernel 3x3
        torch::nn::Conv2d conv( torch::nn::Conv2dOptions( 2, 3, 3 ).stride( 1 ).padding( 1 ) );
        conv->weight = torch::randn( { 3, 2, 3, 3 } ) * 0.1;
        conv->bias = torch::zeros( { 3 } );

        _convNode = std::make_shared<BoundedConvNode>( conv, ConvMode::MATRIX, "test_conv" );
        _convNode->setNodeIndex( 0 );
    }

    std::shared_ptr<BoundedConvNode> _convNode;
};

TEST_F( ConvNodeTest, ForwardPassBasic )
{
    // Input: batch=1, channels=2, height=4, width=4
    torch::Tensor input = torch::randn( { 1, 2, 4, 4 } );
    torch::Tensor output = _convNode->forward( input );

    EXPECT_EQ( output.size( 0 ), 1 ); // Batch
    EXPECT_EQ( output.size( 1 ), 3 ); // Output channels
    EXPECT_GT( output.size( 2 ), 0 ); // Height
    EXPECT_GT( output.size( 3 ), 0 ); // Width
}

TEST_F( ConvNodeTest, IBPBasic )
{
    torch::Tensor lb = torch::zeros( { 1, 2, 4, 4 } );
    torch::Tensor ub = torch::ones( { 1, 2, 4, 4 } );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _convNode->computeIntervalBoundPropagation( inputBounds );

    // Verify bounds computed
    EXPECT_TRUE( result.lower().dim() == 4 || result.lower().dim() == 2 );
    torch::Tensor width = result.upper() - result.lower();
    EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
}

TEST_F( ConvNodeTest, BackwardRejectsAmbiguousFlatSpatialShape )
{
    auto lower = torch::zeros( { 60 } );
    auto upper = torch::ones( { 60 } );
    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lower, upper ) );
    auto A = torch::ones( { 1, 1, 90 } );
    Vector<Pair<BoundA, BoundA>> outputA;
    torch::Tensor lbias;
    torch::Tensor ubias;

    try
    {
        _convNode->boundBackward( BoundA( A ), BoundA( A ), inputBounds, outputA, lbias, ubias );
        FAIL() << "Expected ambiguous flat spatial shape to be rejected";
    }
    catch ( const std::runtime_error &error )
    {
        EXPECT_NE( std::string( error.what() ).find( "ambiguous flat 2D input shape" ),
                   std::string::npos );
    }
}
