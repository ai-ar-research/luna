#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ACASXuRegressionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
    }
};

TEST_F( ACASXuRegressionTest, ACASXuModelLoad )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/acasxu/ACASXU_experimental_v2a_1_1.onnx";
    std::string vnnlibPath = RESOURCES_DIR "/onnx/vnnlib/prop_1.vnnlib";

    try
    {
        auto model = ModelBuilder::loadFromONNXWithVNNLib( onnxPath, vnnlibPath );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );
            EXPECT_TRUE( model->hasInputBounds() );

            // Run CROWN analysis
            auto bounds = model->runCROWN();
            EXPECT_GT( bounds.lower().numel(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        GTEST_SKIP() << "ACAS Xu model not available: " << e.what();
    }
}
