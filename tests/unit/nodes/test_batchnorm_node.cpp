#include <gtest/gtest.h>
#include "src/engine/nodes/BoundedBatchNormNode.h"
#include "src/common/BoundedTensor.h"
#include "src/common/Vector.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class BatchNormNodeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        torch::nn::BatchNorm2d bn( torch::nn::BatchNorm2dOptions( 4 ) );
        auto scale = bn->weight;
        auto bias = bn->bias;
        auto mean = bn->running_mean;
        auto var = bn->running_var;
        float eps = bn->options.eps();
        _bnNode = std::make_shared<BoundedBatchNormNode>( scale, bias, mean, var, eps, "test_bn" );
        _bnNode->setNodeIndex( 0 );
    }

    std::shared_ptr<BoundedBatchNormNode> _bnNode;
};

TEST_F( BatchNormNodeTest, ForwardPassBasic )
{
    torch::Tensor input = torch::randn( { 1, 4, 4, 4 } );
    torch::Tensor output = _bnNode->forward( input );

    EXPECT_EQ( output.sizes(), input.sizes() );
}

TEST_F( BatchNormNodeTest, IBPBasic )
{
    torch::Tensor lb = torch::zeros( { 1, 4, 4, 4 } );
    torch::Tensor ub = torch::ones( { 1, 4, 4, 4 } );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb, ub ) );

    auto result = _bnNode->computeIntervalBoundPropagation( inputBounds );

    torch::Tensor width = result.upper() - result.lower();
    EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
}
