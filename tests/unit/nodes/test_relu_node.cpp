#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedReLUNode.h"
#include "src/engine/nodes/BoundedInputNode.h"
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/engine/TorchModel.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ReLUNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        _reluModule = torch::nn::ReLU();
        _reluNode = std::make_shared<BoundedReLUNode>( _reluModule, "test_relu" );
        _reluNode->setNodeIndex( 0 );
        _reluNode->setInputSize( 10 );
        _reluNode->setOutputSize( 10 );
    }

    torch::nn::ReLU _reluModule;
    std::shared_ptr<BoundedReLUNode> _reluNode;
};

// Forward pass tests
TEST_F( ReLUNodeTest, ForwardPassPositiveInput )
{
    torch::Tensor input = torch::ones( { 1, 10 } );
    torch::Tensor output = _reluNode->forward( input );
    EXPECT_TENSORS_CLOSE( output, input, 1e-5, 1e-4 );
}

TEST_F( ReLUNodeTest, ForwardPassNegativeInput )
{
    torch::Tensor input = -torch::ones( { 1, 10 } );
    torch::Tensor output = _reluNode->forward( input );
    EXPECT_TENSORS_CLOSE( output, torch::zeros( { 1, 10 } ), 1e-5, 1e-4 );
}

TEST_F( ReLUNodeTest, ForwardPassMixedInput )
{
    torch::Tensor input = torch::tensor( { -2.0, -1.0, 0.0, 1.0, 2.0 } ).unsqueeze( 0 );
    torch::Tensor expected = torch::tensor( { 0.0, 0.0, 0.0, 1.0, 2.0 } ).unsqueeze( 0 );
    torch::Tensor output = _reluNode->forward( input );
    EXPECT_TENSORS_CLOSE( output, expected, 1e-5, 1e-4 );
}

// IBP tests
TEST_F( ReLUNodeTest, IBPAlwaysActiveRegion )
{
    torch::Tensor lb = torch::ones( { 1, 10 } ) * 1.0f;
    torch::Tensor ub = torch::ones( { 1, 10 } ) * 2.0f;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _reluNode->computeIntervalBoundPropagation( inputBounds );

    EXPECT_TENSORS_CLOSE( result.lower(), lb, 1e-5, 1e-4 );
    EXPECT_TENSORS_CLOSE( result.upper(), ub, 1e-5, 1e-4 );
}

TEST_F( ReLUNodeTest, IBPAlwaysInactiveRegion )
{
    torch::Tensor lb = -torch::ones( { 1, 10 } ) * 2.0f;
    torch::Tensor ub = -torch::ones( { 1, 10 } ) * 1.0f;

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _reluNode->computeIntervalBoundPropagation( inputBounds );

    EXPECT_TENSORS_CLOSE( result.lower(), torch::zeros( { 1, 10 } ), 1e-5, 1e-4 );
    EXPECT_TENSORS_CLOSE( result.upper(), torch::zeros( { 1, 10 } ), 1e-5, 1e-4 );
}

TEST_F( ReLUNodeTest, IBPUnstableRegion )
{
    // Input spans negative and positive: [-1, 1]
    torch::Tensor lb = -torch::ones( { 1, 10 } );
    torch::Tensor ub = torch::ones( { 1, 10 } );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _reluNode->computeIntervalBoundPropagation( inputBounds );

    // Lower bound should be 0 (inactive part clipped)
    EXPECT_TENSORS_CLOSE( result.lower(), torch::zeros( { 1, 10 } ), 1e-5, 1e-4 );
    // Upper bound should be ub (active part preserved)
    EXPECT_TENSORS_CLOSE( result.upper(), ub, 1e-5, 1e-4 );
}

TEST_F( ReLUNodeTest, IBPBoundWidth )
{
    // Test that bound width doesn't increase (soundness property)
    torch::Tensor lb = torch::tensor( { -0.5, -1.0, 0.0 } ).unsqueeze( 0 );
    torch::Tensor ub = torch::tensor( { 0.5, 1.0, 2.0 } ).unsqueeze( 0 );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _reluNode->computeIntervalBoundPropagation( inputBounds );

    torch::Tensor inputWidth = ub - lb;
    torch::Tensor outputWidth = result.upper() - result.lower();

    // Output width should be <= input width for each element
    auto flatInputWidth = inputWidth.flatten();
    auto flatOutputWidth = outputWidth.flatten();
    for ( int64_t i = 0; i < flatInputWidth.numel(); ++i )
    {
        EXPECT_LE( flatOutputWidth[i].item<float>(), flatInputWidth[i].item<float>() + 1e-5 );
    }
}

TEST_F( ReLUNodeTest, BackwardDenseAFlattensSpatialRelaxationBounds )
{
    const int64_t spec = 2;
    const int64_t batch = 1;
    const int64_t channels = 3;
    const int64_t height = 2;
    const int64_t width = 2;
    const int64_t features = channels * height * width;

    auto lower = -torch::ones( { batch, channels, height, width } );
    auto upper = torch::ones( { batch, channels, height, width } );
    Vector<BoundedTensor<torch::Tensor>> input_bounds;
    input_bounds.append( BoundedTensor<torch::Tensor>( lower, upper ) );

    auto dense_A = torch::ones( { spec, batch, features } );
    Vector<Pair<BoundA, BoundA>> output_A;
    torch::Tensor lower_bias;
    torch::Tensor upper_bias;

    ASSERT_NO_THROW( _reluNode->boundBackward( BoundA( dense_A ), BoundA( dense_A ), input_bounds,
                                               output_A, lower_bias, upper_bias ) );
    ASSERT_EQ( output_A.size(), 1u );
    ASSERT_TRUE( output_A[0].first().isTensor() );
    ASSERT_TRUE( output_A[0].second().isTensor() );
    EXPECT_EQ( output_A[0].first().asTensor().sizes(), dense_A.sizes() );
    EXPECT_EQ( output_A[0].second().asTensor().sizes(), dense_A.sizes() );
    EXPECT_EQ( lower_bias.sizes(), torch::IntArrayRef( { spec, batch } ) );
    EXPECT_EQ( upper_bias.sizes(), torch::IntArrayRef( { spec, batch } ) );
}

TEST_F( ReLUNodeTest, BackwardSpatialDenseAReshapesFlatPerSpecAlpha )
{
    const int64_t spec = 2;
    const int64_t batch = 1;
    const int64_t channels = 3;
    const int64_t height = 2;
    const int64_t width = 2;
    const int64_t features = channels * height * width;

    auto input_node = std::make_shared<BoundedInputNode>( 0, features, "input" );
    auto relu_node = std::make_shared<BoundedReLUNode>( torch::nn::ReLU(), "relu" );
    input_node->setNodeIndex( 0 );
    relu_node->setNodeIndex( 1 );
    relu_node->setInputSize( features );
    relu_node->setOutputSize( features );

    Vector<std::shared_ptr<BoundedTorchNode>> nodes;
    nodes.append( input_node );
    nodes.append( relu_node );
    Vector<unsigned> input_indices;
    input_indices.append( 0 );
    Map<unsigned, Vector<unsigned>> dependencies;
    dependencies[0] = Vector<unsigned>();
    dependencies[1] = Vector<unsigned>( { 0 } );
    TorchModel model( nodes, input_indices, 1, dependencies );
    NLR::AlphaCROWNAnalysis alpha_analysis( &model );
    alpha_analysis.getCROWNAnalysis()->_setCurrentStart( "spatial_relu", spec );
    relu_node->setAlphaCrownAnalysis( &alpha_analysis );
    relu_node->setOptimizationStage( "opt" );

    auto lower = -torch::ones( { batch, channels, height, width } );
    auto upper = torch::ones( { batch, channels, height, width } );
    Vector<BoundedTensor<torch::Tensor>> input_bounds;
    input_bounds.append( BoundedTensor<torch::Tensor>( lower, upper ) );
    auto spatial_A = torch::ones( { spec, batch, channels, height, width } );
    Vector<Pair<BoundA, BoundA>> output_A;
    torch::Tensor lower_bias;
    torch::Tensor upper_bias;

    ASSERT_NO_THROW( relu_node->boundBackward( BoundA( spatial_A ), BoundA( spatial_A ),
                                               input_bounds, output_A, lower_bias, upper_bias ) );
    ASSERT_EQ( output_A.size(), 1u );
    EXPECT_EQ( output_A[0].first().asTensor().sizes(), spatial_A.sizes() );
    EXPECT_EQ( output_A[0].second().asTensor().sizes(), spatial_A.sizes() );
    EXPECT_EQ( lower_bias.sizes(), torch::IntArrayRef( { spec, batch } ) );
    EXPECT_EQ( upper_bias.sizes(), torch::IntArrayRef( { spec, batch } ) );
}
