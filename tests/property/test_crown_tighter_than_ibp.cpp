#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

TEST( CROWNTighterThanIBPProperty, CROWNAlwaysTighter )
{
    torch::manual_seed( 42 );
    at::set_num_threads( 1 );
    at::set_num_interop_threads( 1 );

    auto model = ModelBuilder::createMLP( 4, { 6 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 4 }, 0.0, 1.0, 0.2 );

    model->setInputBounds( inputBounds );

    // Get IBP bounds
    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    // Get CROWN bounds
    auto crownBounds = model->runCROWN();

    // CROWN should always be at least as tight as IBP
    EXPECT_TRUE( SoundnessChecker::crownTighterThanIBP( ibpBounds, crownBounds ) );
}
