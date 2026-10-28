#include <gtest/gtest.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>
#include <random>

using namespace NLR;
using namespace test;

TEST( RandomNetworksProperty, FuzzTest )
{
    torch::manual_seed( 42 );
    at::set_num_threads( 1 );
    at::set_num_interop_threads( 1 );

    std::mt19937 gen( 12345 );
    std::uniform_int_distribution<> sizeDist( 2, 10 );
    std::uniform_int_distribution<> depthDist( 1, 3 );

    // Test multiple random architectures
    for ( int trial = 0; trial < 5; ++trial )
    {
        unsigned inputSize = sizeDist( gen );
        unsigned outputSize = sizeDist( gen );
        unsigned depth = depthDist( gen );

        std::vector<unsigned> hiddenSizes;
        for ( unsigned i = 0; i < depth; ++i )
        {
            hiddenSizes.push_back( sizeDist( gen ) );
        }

        auto model = ModelBuilder::createMLP( inputSize, hiddenSizes, outputSize, true, true );

        auto inputBounds = BoundGenerator::randomBounds( { 1, inputSize }, -1.0, 1.0, 0.3 );

        model->setInputBounds( inputBounds );

        // Verify analysis runs without error
        EXPECT_NO_THROW( {
            auto bounds = model->runCROWN();
            EXPECT_GT( bounds.lower().numel(), 0 );
        } );
    }
}
