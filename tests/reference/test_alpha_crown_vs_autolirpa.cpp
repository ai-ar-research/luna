#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class AlphaCROWNVsAutoLiRPATest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
        LunaConfiguration::ALPHA_ITERATIONS = 10;
    }
};

TEST_F( AlphaCROWNVsAutoLiRPATest, CompareAlphaCROWNBounds )
{
    auto model = ModelBuilder::createMLP( 4, { 6 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 4 } ), 0.1 );

    model->setInputBounds( inputBounds );
    auto cppBounds = model->runAlphaCROWN( true, false );

    // TODO: compare against Python auto_LiRPA reference bounds.
    EXPECT_GT( cppBounds.lower().numel(), 0 );
    EXPECT_GT( cppBounds.upper().numel(), 0 );
}
