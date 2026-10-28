#include <gtest/gtest.h>
#include "src/configuration/LunaConfiguration.h"
#include "src/engine/CROWNAnalysis.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace test;

TEST( GPUDeviceTest, CROWNUsesConfiguredCudaDevice )
{
    if ( !torch::cuda::is_available() )
    {
        return;
    }

    LunaConfiguration::USE_CUDA = true;
    LunaConfiguration::CUDA_DEVICE_ID = 0;
    LunaConfiguration::updateDeviceFromFlags();

    auto model = ModelBuilder::createMLP( 4, { 8 }, 2, true, false );
    auto device = LunaConfiguration::getDevice();

    torch::Tensor lower =
        torch::zeros( { 1, 4 }, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );
    torch::Tensor upper =
        torch::ones( { 1, 4 }, torch::TensorOptions().dtype( torch::kFloat32 ).device( device ) );
    BoundedTensor<torch::Tensor> inputBounds( lower, upper );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.run();

    auto bounds = analysis.getOutputBounds();
    EXPECT_TRUE( bounds.lower().is_cuda() );
    EXPECT_TRUE( bounds.upper().is_cuda() );
}
