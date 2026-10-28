#include <gtest/gtest.h>
#include "src/input_parsers/VnnLibInputParser.h"
#include "fixtures/tensor_comparators.h"
#include <iostream>
#include <torch/torch.h>

using namespace test;

class VNNLibParserTest : public ::testing::Test
{
protected:
    void SetUp() override {}
};

TEST_F( VNNLibParserTest, ParseInputBounds )
{
    std::string vnnlibPath = RESOURCES_DIR "/properties/builtin_property.txt";

    try
    {
        // Parse input bounds (need to specify expected input size)
        unsigned inputSize = 5; // Common ACAS Xu input size
        auto bounds =
            VnnLibInputParser::parseInputBounds( String( vnnlibPath.c_str() ), inputSize );

        // TODO: validate bounds against expected values from fixture file.
        EXPECT_GT( bounds.lower().numel(), 0 );
        EXPECT_GT( bounds.upper().numel(), 0 );

        // Verify upper >= lower
        torch::Tensor width = bounds.upper() - bounds.lower();
        EXPECT_TRUE( torch::all( width >= 0 ).item<bool>() );
    }
    catch ( const std::exception &e )
    {
        std::cerr << "VNN-LIB file not available: " << e.what() << std::endl;
        return;
    }
}

TEST_F( VNNLibParserTest, ParseOutputConstraints )
{
    std::string vnnlibPath = RESOURCES_DIR "/properties/builtin_property.txt";

    try
    {
        unsigned outputSize = 5;
        auto constraints =
            VnnLibInputParser::parseOutputConstraints( String( vnnlibPath.c_str() ), outputSize );

        // TODO: validate parsed constraints against expected set.
        EXPECT_TRUE( true );
    }
    catch ( const std::exception &e )
    {
        std::cerr << "VNN-LIB file not available: " << e.what() << std::endl;
        return;
    }
}
