#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedLinearNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class LinearNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a 2x3 linear layer: input size 2, output size 3
        torch::nn::Linear linear( torch::nn::LinearOptions( 2, 3 ) );
        linear->weight =
            torch::tensor( { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } }, torch::kFloat32 );
        linear->bias = torch::tensor( { 0.1, 0.2, 0.3 }, torch::kFloat32 );

        _linearNode = std::make_shared<BoundedLinearNode>( linear, 1.0f, "test_linear" );
        _linearNode->setNodeIndex( 0 );
        _linearNode->setInputSize( 2 );
        _linearNode->setOutputSize( 3 );
    }

    std::shared_ptr<BoundedLinearNode> _linearNode;
};

// Forward pass tests
TEST_F( LinearNodeTest, ForwardPassSimple )
{
    torch::Tensor input = torch::tensor( { 1.0, 2.0 } ).unsqueeze( 0 );
    torch::Tensor output = _linearNode->forward( input );

    // Expected: weight @ input + bias
    // [1,2] @ [1;2] + [0.1;0.2;0.3] = [5;11;17] + [0.1;0.2;0.3] = [5.1;11.2;17.3]
    torch::Tensor expected = torch::tensor( { 5.1, 11.2, 17.3 } ).unsqueeze( 0 );
    EXPECT_TENSORS_CLOSE( output, expected, 1e-5, 1e-4 );
}

TEST_F( LinearNodeTest, ForwardPassBatch )
{
    torch::Tensor input = torch::tensor( { { 1.0, 2.0 }, { 3.0, 4.0 } }, torch::kFloat32 );
    torch::Tensor output = _linearNode->forward( input );

    EXPECT_EQ( output.size( 0 ), 2 ); // Batch size
    EXPECT_EQ( output.size( 1 ), 3 ); // Output size
}

// IBP tests
TEST_F( LinearNodeTest, IBPBasic )
{
    torch::Tensor lb = torch::tensor( { 0.0, 0.0 } ).unsqueeze( 0 );
    torch::Tensor ub = torch::tensor( { 1.0, 1.0 } ).unsqueeze( 0 );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _linearNode->computeIntervalBoundPropagation( inputBounds );

    // Verify output bounds are computed
    EXPECT_EQ( result.lower().size( 1 ), 3 );
    EXPECT_EQ( result.upper().size( 1 ), 3 );

    // Verify upper >= lower for all outputs
    torch::Tensor width = result.upper() - result.lower();
    EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
}

TEST_F( LinearNodeTest, IBPConsistentWithForward )
{
    // Test that forward pass values are within IBP bounds
    torch::Tensor center = torch::tensor( { 0.5, 0.5 } ).unsqueeze( 0 );
    torch::Tensor output = _linearNode->forward( center );

    torch::Tensor lb = torch::tensor( { 0.0, 0.0 } ).unsqueeze( 0 );
    torch::Tensor ub = torch::tensor( { 1.0, 1.0 } ).unsqueeze( 0 );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _linearNode->computeIntervalBoundPropagation( inputBounds );

    // Check that center output is within bounds
    for ( int64_t i = 0; i < output.size( 1 ); ++i )
    {
        EXPECT_GE( output[0][i].item<float>(), result.lower()[0][i].item<float>() - 1e-5 );
        EXPECT_LE( output[0][i].item<float>(), result.upper()[0][i].item<float>() + 1e-5 );
    }
}
