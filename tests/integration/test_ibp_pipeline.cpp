#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

TEST(IBPPipelineTest, PublishesIntermediateBounds)
{
    auto model = ModelBuilder::createMLP(2, {2}, 1, true, false);
    auto inputBounds = BoundedTensor<torch::Tensor>(
        torch::tensor({-1.0f, 0.0f}), torch::tensor({1.0f, 2.0f}));

    auto outputBounds = model->runIBP(inputBounds);

    ASSERT_TRUE(model->hasConcreteBounds(1));
    const auto hiddenBounds = model->getConcreteBounds(1);
    EXPECT_EQ(hiddenBounds.lower().numel(), 2);
    EXPECT_EQ(hiddenBounds.upper().numel(), 2);
    EXPECT_EQ(outputBounds.lower().numel(), 1);
    EXPECT_EQ(outputBounds.upper().numel(), 1);
}
