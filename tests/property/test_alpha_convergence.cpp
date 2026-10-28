#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

TEST( AlphaConvergenceProperty, AlphaCROWNImprovesOrMaintains )
{
    torch::manual_seed( 42 );
    at::set_num_threads( 1 );
    at::set_num_interop_threads( 1 );

    LunaConfiguration::ALPHA_ITERATIONS = 5;

    auto model = ModelBuilder::createMLP( 5, { 8 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );

    // Get CROWN bounds
    auto crownBounds = model->runCROWN();

    // Get AlphaCROWN bounds
    auto alphaBounds = model->runAlphaCROWN( true, false );

    // AlphaCROWN should be at least as tight
    EXPECT_TRUE( SoundnessChecker::alphaCrownTighterThanCrown( crownBounds, alphaBounds ) );
}
