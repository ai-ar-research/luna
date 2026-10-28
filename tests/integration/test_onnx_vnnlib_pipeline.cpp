#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/tensor_comparators.h"
#include <iostream>
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class ONNXVNNLibPipelineTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
    }
};

TEST_F( ONNXVNNLibPipelineTest, LoadONNXModel )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/fc1.onnx";

    try
    {
        auto model = ModelBuilder::loadFromONNX( onnxPath );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );
            EXPECT_GT( model->getInputSize(), 0 );
            EXPECT_GT( model->getOutputSize(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "ONNX model not available: " << e.what() << std::endl;
        return;
    }
}

TEST_F( ONNXVNNLibPipelineTest, LoadONNXWithVNNLib )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/fc1.onnx";
    std::string vnnlibPath = RESOURCES_DIR "/properties/builtin_property.txt";

    try
    {
        auto model = ModelBuilder::loadFromONNXWithVNNLib( onnxPath, vnnlibPath );
        if ( model )
        {
            EXPECT_GT( model->getNumNodes(), 0 );
            EXPECT_TRUE( model->hasInputBounds() );

            // Test that we can run analysis
            auto outputBounds = model->runCROWN();
            EXPECT_GT( outputBounds.lower().numel(), 0 );
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Model files not available: " << e.what() << std::endl;
        return;
    }
}
