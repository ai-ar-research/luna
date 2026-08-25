#include <gtest/gtest.h>
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/conv/Patches.h"
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include "fixtures/tensor_comparators.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

class CROWNAnalysisTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Use deterministic seed for reproducibility
        torch::manual_seed( 42 );
        static bool threadsConfigured = false;
        if ( !threadsConfigured )
        {
            at::set_num_threads( 1 );
            at::set_num_interop_threads( 1 );
            threadsConfigured = true;
        }
    }
};

TEST_F( CROWNAnalysisTest, IBPForwardBasic )
{
    // Create a simple MLP: Input(2) -> Linear(2,3) -> ReLU -> Linear(3,2) -> Output
    auto model = ModelBuilder::createMLP( 2, { 3 }, 2, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::zeros( { 1, 2 } ), 0.1 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();

    auto ibpBounds = analysis.getOutputIBPBounds();

    // Verify bounds are computed
    EXPECT_GT( ibpBounds.lower().numel(), 0 );
    EXPECT_GT( ibpBounds.upper().numel(), 0 );

    // Verify upper >= lower
    torch::Tensor width = ibpBounds.upper() - ibpBounds.lower();
    EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
}

TEST_F( CROWNAnalysisTest, CROWNTighterThanIBP )
{
    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.2 );
    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.computeIBPBounds();
    auto ibpBounds = analysis.getOutputIBPBounds();

    analysis.run();
    auto crownBounds = analysis.getOutputBounds();

    // CROWN should be at least as tight as IBP
    EXPECT_TRUE( SoundnessChecker::crownTighterThanIBP( ibpBounds, crownBounds ) );
}

TEST_F( CROWNAnalysisTest, CROWNSoundnessSimple )
{
    auto model = ModelBuilder::createMLP( 5, { 10 }, 3, true, false );

    auto inputBounds = BoundGenerator::epsilonBall( torch::randn( { 1, 5 } ), 0.1 );

    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.run();

    auto outputBounds = analysis.getOutputBounds();

    // Verify soundness by checking center point
    EXPECT_TRUE( SoundnessChecker::centerContainedInBounds( inputBounds, outputBounds, *model ) );
}

TEST_F( CROWNAnalysisTest, BackwardPropagation )
{
    auto model = ModelBuilder::createMLP( 3, { 5 }, 2, true, false );

    auto inputBounds = BoundGenerator::wideBounds( { 1, 3 }, 0.0, 1.0 );
    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );
    analysis.run();

    // Verify bounds computed successfully
    auto bounds = analysis.getOutputBounds();
    EXPECT_GT( bounds.lower().numel(), 0 );
    EXPECT_GT( bounds.upper().numel(), 0 );
}

TEST_F( CROWNAnalysisTest, AddAMixedPatchesAndSpatialDenseUsesCanonicalLayout )
{
    auto model = ModelBuilder::createMLP( 4, {}, 4, false, false );
    NLR::CROWNAnalysis analysis( model.get() );

    auto patch_values = torch::arange( 1, 5, torch::kFloat32 ).reshape( { 1, 1, 2, 2, 1, 1, 1 } );
    auto patches = std::make_shared<Patches>(
        patch_values, std::vector<int64_t>{ 1, 1 }, std::vector<int64_t>{ 0, 0, 0, 0 },
        std::vector<int64_t>{ 0, 0, 0, 0 }, 0, std::nullopt, std::vector<int64_t>{ 1, 1, 2, 2 },
        std::vector<int64_t>{ 1, 1, 2, 2 } );

    auto patch_matrix =
        patches->to_matrix( patches->input_shape ).permute( { 1, 0, 2, 3, 4 } ).flatten( 2 );
    auto spatial_dense = patch_matrix.reshape( { 4, 1, 1, 2, 2 } );
    auto expected = patch_matrix * 2;

    auto patches_first = analysis.addA( BoundA( patches ), BoundA( spatial_dense ) );
    auto dense_first = analysis.addA( BoundA( spatial_dense ), BoundA( patches ) );

    ASSERT_TRUE( patches_first.isTensor() );
    ASSERT_TRUE( dense_first.isTensor() );
    EXPECT_EQ( patches_first.asTensor().sizes(), torch::IntArrayRef( { 4, 1, 4 } ) );
    EXPECT_EQ( dense_first.asTensor().sizes(), torch::IntArrayRef( { 4, 1, 4 } ) );
    EXPECT_TRUE( torch::allclose( patches_first.asTensor(), expected ) );
    EXPECT_TRUE( torch::allclose( dense_first.asTensor(), expected ) );
}

TEST_F( CROWNAnalysisTest, AddAMixedPatchesAndDenseRejectsIncompatibleFeaturesClearly )
{
    auto model = ModelBuilder::createMLP( 4, {}, 4, false, false );
    NLR::CROWNAnalysis analysis( model.get() );

    auto patches = std::make_shared<Patches>(
        torch::ones( { 1, 1, 2, 2, 1, 1, 1 } ), std::vector<int64_t>{ 1, 1 },
        std::vector<int64_t>{ 0, 0, 0, 0 }, std::vector<int64_t>{ 0, 0, 0, 0 }, 0, std::nullopt,
        std::vector<int64_t>{ 1, 1, 2, 2 }, std::vector<int64_t>{ 1, 1, 2, 2 } );
    auto incompatible_dense = torch::ones( { 4, 1, 5 } );

    try
    {
        (void)analysis.addA( BoundA( patches ), BoundA( incompatible_dense ) );
        FAIL() << "Expected incompatible mixed A matrices to throw";
    }
    catch ( const std::runtime_error &error )
    {
        const std::string message = error.what();
        EXPECT_NE( message.find( "CROWNAnalysis::addA" ), std::string::npos );
        EXPECT_NE( message.find( "Patches" ), std::string::npos );
        EXPECT_NE( message.find( "[4, 1, 4]" ), std::string::npos );
        EXPECT_NE( message.find( "[4, 1, 5]" ), std::string::npos );
    }
}
