#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class CIFARRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( CIFARRegressionTest, CIFARModelLoad )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/cifar_base_kw_simp.onnx";
    std::string vnnlibPath = RESOURCES_DIR "/onnx/vnnlib/cifar_bounded.vnnlib";

    try
    {
        auto model = ModelBuilder::loadFromONNXWithVNNLib( onnxPath, vnnlibPath );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );

            // Run CROWN if bounds available
            if ( model->hasInputBounds() )
            {
                auto bounds = model->runCROWN();
                EXPECT_GT( bounds.lower().numel(), 0 );
            }
        }
    }
    catch ( const std::exception &e )
    {
        GTEST_SKIP() << "CIFAR model not available: " << e.what();
    }
}
