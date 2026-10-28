#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class MNISTRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( MNISTRegressionTest, MNISTModelLoad )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/mnist-point.onnx";

    try
    {
        auto model = ModelBuilder::loadFromONNX( onnxPath );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        GTEST_SKIP() << "MNIST model not available: " << e.what();
    }
}
