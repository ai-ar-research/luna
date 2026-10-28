#include "src/input_parsers/OnnxToTorch.h"
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/nodes/BoundedConvNode.h"
#include "src/engine/conv/Patches.h"
#include "src/engine/conv/MatrixConvolution.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <map>

using namespace NLR;

// Structure to hold diagnostic information
struct ConvDiagnostics
{
    std::string mode;
    std::string layer_name;

    // Tensor shapes
    std::vector<int64_t> input_shape;
    std::vector<int64_t> weight_shape;
    std::vector<int64_t> output_shape;

    // Backward pass shapes
    std::vector<int64_t> last_A_shape;
    std::vector<int64_t> next_A_shape;
    std::vector<int64_t> bias_contrib_shape;

    // Patches specific
    std::vector<int64_t> patches_shape;
    std::vector<int64_t> patches_stride;
    std::vector<int64_t> patches_padding;
    int64_t patches_inserted_zeros;

    // Matrix mode specific
    std::vector<int64_t> im2col_shape;
    std::vector<int64_t> weight_matrix_shape;

    // Memory usage
    int64_t input_memory_bytes;
    int64_t output_memory_bytes;
    int64_t intermediate_memory_bytes;
    int64_t total_memory_bytes;

    // Performance metrics
    double forward_time_ms;
    double backward_time_ms;
};

// Enhanced BoundedConvNode for diagnostics
class DiagnosticConvNode : public BoundedConvNode
{
public:
    ConvDiagnostics diagnostics;

    DiagnosticConvNode( const torch::nn::Conv2d &convModule, ConvMode mode, const String &name )
        : BoundedConvNode( convModule, mode, name )
        , convModule_( convModule )
    {
        diagnostics.mode = ( mode == ConvMode::MATRIX ) ? "matrix" : "patches";
        diagnostics.layer_name = std::string( name.ascii() );
    }

private:
    torch::nn::Conv2d convModule_;

public:
    torch::Tensor forward( const torch::Tensor &input ) override
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Record input shape
        diagnostics.input_shape.clear();
        for ( int i = 0; i < input.dim(); ++i )
        {
            diagnostics.input_shape.push_back( input.size( i ) );
        }
        diagnostics.input_memory_bytes = input.numel() * sizeof( float );

        // Call parent forward
        torch::Tensor output = BoundedConvNode::forward( input );

        // Record output shape
        diagnostics.output_shape.clear();
        for ( int i = 0; i < output.dim(); ++i )
        {
            diagnostics.output_shape.push_back( output.size( i ) );
        }
        diagnostics.output_memory_bytes = output.numel() * sizeof( float );

        // Record weight shape
        if ( convModule_ && convModule_->weight.defined() )
        {
            diagnostics.weight_shape.clear();
            for ( int i = 0; i < convModule_->weight.dim(); ++i )
            {
                diagnostics.weight_shape.push_back( convModule_->weight.size( i ) );
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        diagnostics.forward_time_ms =
            std::chrono::duration<double, std::milli>( end - start ).count();

        return output;
    }

    void boundBackward( const BoundA &last_lA, const BoundA &last_uA,
                        const Vector<BoundedTensor<torch::Tensor>> &inputBounds,
                        Vector<Pair<BoundA, BoundA>> &outputA_matrices, torch::Tensor &lbias,
                        torch::Tensor &ubias ) override
    {

        auto start = std::chrono::high_resolution_clock::now();

        // Record last_A shape
        if ( last_lA.defined() )
        {
            diagnostics.last_A_shape.clear();
            if ( last_lA.isTensor() )
            {
                auto tensor = last_lA.asTensor();
                for ( int i = 0; i < tensor.dim(); ++i )
                {
                    diagnostics.last_A_shape.push_back( tensor.size( i ) );
                }
            }
            else if ( last_lA.isPatches() )
            {
                auto patches = last_lA.asPatches();
                for ( int i = 0; i < patches->patches.dim(); ++i )
                {
                    diagnostics.patches_shape.push_back( patches->patches.size( i ) );
                }
                diagnostics.patches_stride = patches->stride;
                diagnostics.patches_padding = patches->padding;
                diagnostics.patches_inserted_zeros = patches->inserted_zeros;
            }
        }

        // Call parent backward
        BoundedConvNode::boundBackward( last_lA, last_uA, inputBounds, outputA_matrices, lbias,
                                        ubias );

        // Record next_A shape
        if ( !outputA_matrices.empty() && outputA_matrices[0].first().defined() )
        {
            diagnostics.next_A_shape.clear();
            if ( outputA_matrices[0].first().isTensor() )
            {
                auto tensor = outputA_matrices[0].first().asTensor();
                for ( int i = 0; i < tensor.dim(); ++i )
                {
                    diagnostics.next_A_shape.push_back( tensor.size( i ) );
                }
                diagnostics.intermediate_memory_bytes = tensor.numel() * sizeof( float );
            }
        }

        // Record bias contribution shape
        if ( lbias.defined() )
        {
            diagnostics.bias_contrib_shape.clear();
            for ( int i = 0; i < lbias.dim(); ++i )
            {
                diagnostics.bias_contrib_shape.push_back( lbias.size( i ) );
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        diagnostics.backward_time_ms =
            std::chrono::duration<double, std::milli>( end - start ).count();

        // Calculate total memory
        diagnostics.total_memory_bytes = diagnostics.input_memory_bytes +
                                         diagnostics.output_memory_bytes +
                                         diagnostics.intermediate_memory_bytes;
    }

    void printDiagnostics()
    {
        std::cout << "\n=== Convolution Diagnostics [" << diagnostics.layer_name << " - "
                  << diagnostics.mode << " mode] ===" << std::endl;

        std::cout << "Input shape: [";
        for ( size_t i = 0; i < diagnostics.input_shape.size(); ++i )
        {
            if ( i > 0 )
                std::cout << ", ";
            std::cout << diagnostics.input_shape[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "Weight shape: [";
        for ( size_t i = 0; i < diagnostics.weight_shape.size(); ++i )
        {
            if ( i > 0 )
                std::cout << ", ";
            std::cout << diagnostics.weight_shape[i];
        }
        std::cout << "]" << std::endl;

        std::cout << "Output shape: [";
        for ( size_t i = 0; i < diagnostics.output_shape.size(); ++i )
        {
            if ( i > 0 )
                std::cout << ", ";
            std::cout << diagnostics.output_shape[i];
        }
        std::cout << "]" << std::endl;

        if ( !diagnostics.last_A_shape.empty() )
        {
            std::cout << "Last_A shape: [";
            for ( size_t i = 0; i < diagnostics.last_A_shape.size(); ++i )
            {
                if ( i > 0 )
                    std::cout << ", ";
                std::cout << diagnostics.last_A_shape[i];
            }
            std::cout << "]" << std::endl;
        }

        if ( !diagnostics.next_A_shape.empty() )
        {
            std::cout << "Next_A shape: [";
            for ( size_t i = 0; i < diagnostics.next_A_shape.size(); ++i )
            {
                if ( i > 0 )
                    std::cout << ", ";
                std::cout << diagnostics.next_A_shape[i];
            }
            std::cout << "]" << std::endl;
        }

        if ( !diagnostics.patches_shape.empty() )
        {
            std::cout << "Patches shape: [";
            for ( size_t i = 0; i < diagnostics.patches_shape.size(); ++i )
            {
                if ( i > 0 )
                    std::cout << ", ";
                std::cout << diagnostics.patches_shape[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "Patches stride: [" << diagnostics.patches_stride[0] << ", "
                      << diagnostics.patches_stride[1] << "]" << std::endl;
            std::cout << "Patches padding: [";
            for ( size_t i = 0; i < diagnostics.patches_padding.size(); ++i )
            {
                if ( i > 0 )
                    std::cout << ", ";
                std::cout << diagnostics.patches_padding[i];
            }
            std::cout << "]" << std::endl;
            std::cout << "Patches inserted_zeros: " << diagnostics.patches_inserted_zeros
                      << std::endl;
        }

        std::cout << "\nMemory Usage:" << std::endl;
        std::cout << "  Input: " << diagnostics.input_memory_bytes / 1024.0 << " KB" << std::endl;
        std::cout << "  Output: " << diagnostics.output_memory_bytes / 1024.0 << " KB" << std::endl;
        std::cout << "  Intermediate: " << diagnostics.intermediate_memory_bytes / 1024.0 << " KB"
                  << std::endl;
        std::cout << "  Total: " << diagnostics.total_memory_bytes / 1024.0 << " KB" << std::endl;

        std::cout << "\nTiming:" << std::endl;
        std::cout << "  Forward: " << diagnostics.forward_time_ms << " ms" << std::endl;
        std::cout << "  Backward: " << diagnostics.backward_time_ms << " ms" << std::endl;
    }

    void exportToJSON( const std::string &filename )
    {
        std::ofstream file( filename );
        file << "{\n";
        file << "  \"mode\": \"" << diagnostics.mode << "\",\n";
        file << "  \"layer_name\": \"" << diagnostics.layer_name << "\",\n";

        // Export shapes
        file << "  \"input_shape\": [";
        for ( size_t i = 0; i < diagnostics.input_shape.size(); ++i )
        {
            if ( i > 0 )
                file << ", ";
            file << diagnostics.input_shape[i];
        }
        file << "],\n";

        file << "  \"weight_shape\": [";
        for ( size_t i = 0; i < diagnostics.weight_shape.size(); ++i )
        {
            if ( i > 0 )
                file << ", ";
            file << diagnostics.weight_shape[i];
        }
        file << "],\n";

        file << "  \"output_shape\": [";
        for ( size_t i = 0; i < diagnostics.output_shape.size(); ++i )
        {
            if ( i > 0 )
                file << ", ";
            file << diagnostics.output_shape[i];
        }
        file << "],\n";

        file << "  \"last_A_shape\": [";
        for ( size_t i = 0; i < diagnostics.last_A_shape.size(); ++i )
        {
            if ( i > 0 )
                file << ", ";
            file << diagnostics.last_A_shape[i];
        }
        file << "],\n";

        file << "  \"next_A_shape\": [";
        for ( size_t i = 0; i < diagnostics.next_A_shape.size(); ++i )
        {
            if ( i > 0 )
                file << ", ";
            file << diagnostics.next_A_shape[i];
        }
        file << "],\n";

        if ( !diagnostics.patches_shape.empty() )
        {
            file << "  \"patches_shape\": [";
            for ( size_t i = 0; i < diagnostics.patches_shape.size(); ++i )
            {
                if ( i > 0 )
                    file << ", ";
                file << diagnostics.patches_shape[i];
            }
            file << "],\n";

            file << "  \"patches_stride\": [" << diagnostics.patches_stride[0] << ", "
                 << diagnostics.patches_stride[1] << "],\n";

            file << "  \"patches_padding\": [";
            for ( size_t i = 0; i < diagnostics.patches_padding.size(); ++i )
            {
                if ( i > 0 )
                    file << ", ";
                file << diagnostics.patches_padding[i];
            }
            file << "],\n";

            file << "  \"patches_inserted_zeros\": " << diagnostics.patches_inserted_zeros << ",\n";
        }

        // Memory and timing
        file << "  \"memory_bytes\": {\n";
        file << "    \"input\": " << diagnostics.input_memory_bytes << ",\n";
        file << "    \"output\": " << diagnostics.output_memory_bytes << ",\n";
        file << "    \"intermediate\": " << diagnostics.intermediate_memory_bytes << ",\n";
        file << "    \"total\": " << diagnostics.total_memory_bytes << "\n";
        file << "  },\n";

        file << "  \"timing_ms\": {\n";
        file << "    \"forward\": " << diagnostics.forward_time_ms << ",\n";
        file << "    \"backward\": " << diagnostics.backward_time_ms << "\n";
        file << "  }\n";

        file << "}\n";
        file.close();

        std::cout << "Diagnostics exported to " << filename << std::endl;
    }
};

// Test function to run diagnostics on various configurations
void runDiagnosticTest( const std::string &test_name, int in_channels, int out_channels,
                        int kernel_size, int stride, int padding, int input_height, int input_width,
                        int batch_size = 1 )
{

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test: " << test_name << std::endl;
    std::cout << "========================================" << std::endl;

    // Create Conv2d module
    auto conv_options = torch::nn::Conv2dOptions( in_channels, out_channels, kernel_size )
                            .stride( stride )
                            .padding( padding );
    torch::nn::Conv2d conv_module( conv_options );

    // Initialize weights
    torch::nn::init::xavier_uniform_( conv_module->weight );
    if ( conv_module->bias.defined() )
    {
        torch::nn::init::zeros_( conv_module->bias );
    }

    // Create diagnostic nodes for both modes
    DiagnosticConvNode conv_matrix( conv_module, ConvMode::MATRIX, test_name + "_matrix" );
    DiagnosticConvNode conv_patches( conv_module, ConvMode::PATCHES, test_name + "_patches" );

    // Create input
    torch::Tensor input = torch::randn( { batch_size, in_channels, input_height, input_width } );

    std::cout << "\n--- Matrix Mode ---" << std::endl;
    torch::Tensor output_matrix = conv_matrix.forward( input );

    std::cout << "\n--- Patches Mode ---" << std::endl;
    torch::Tensor output_patches = conv_patches.forward( input );

    // Test backward pass
    int output_size = output_matrix.numel() / batch_size;
    torch::Tensor last_A_tensor = torch::randn( { batch_size, output_size } );
    BoundA last_A( last_A_tensor );

    Vector<BoundedTensor<torch::Tensor>> inputBounds;
    torch::Tensor lb = input - 0.1;
    torch::Tensor ub = input + 0.1;
    inputBounds.append( BoundedTensor<torch::Tensor>( lb.flatten(), ub.flatten() ) );

    Vector<Pair<BoundA, BoundA>> outputA_matrix, outputA_patches;
    torch::Tensor lbias_matrix, ubias_matrix, lbias_patches, ubias_patches;

    std::cout << "\n--- Matrix Mode Backward ---" << std::endl;
    conv_matrix.boundBackward( last_A, last_A, inputBounds, outputA_matrix, lbias_matrix,
                               ubias_matrix );

    std::cout << "\n--- Patches Mode Backward ---" << std::endl;
    conv_patches.boundBackward( last_A, last_A, inputBounds, outputA_patches, lbias_patches,
                                ubias_patches );

    // Print diagnostics
    conv_matrix.printDiagnostics();
    conv_patches.printDiagnostics();

    // Export to JSON for comparison
    conv_matrix.exportToJSON( test_name + "_matrix.json" );
    conv_patches.exportToJSON( test_name + "_patches.json" );

    // Compare outputs
    std::cout << "\n--- Comparison ---" << std::endl;
    torch::Tensor diff = torch::abs( output_matrix - output_patches );
    float max_diff = diff.max().item<float>();
    float mean_diff = diff.mean().item<float>();

    std::cout << "Forward pass difference:" << std::endl;
    std::cout << "  Max: " << max_diff << std::endl;
    std::cout << "  Mean: " << mean_diff << std::endl;

    if ( max_diff < 1e-5 )
    {
        std::cout << "  ✓ Outputs match!" << std::endl;
    }
    else
    {
        std::cout << "  ✗ Outputs differ significantly!" << std::endl;
    }

    // Memory comparison
    float memory_ratio = (float)conv_patches.diagnostics.total_memory_bytes /
                         (float)conv_matrix.diagnostics.total_memory_bytes;
    std::cout << "\nMemory usage ratio (patches/matrix): " << memory_ratio << std::endl;

    // Speed comparison
    float forward_speedup =
        conv_matrix.diagnostics.forward_time_ms / conv_patches.diagnostics.forward_time_ms;
    float backward_speedup =
        conv_matrix.diagnostics.backward_time_ms / conv_patches.diagnostics.backward_time_ms;

    std::cout << "Speed comparison:" << std::endl;
    std::cout << "  Forward speedup (patches vs matrix): " << forward_speedup << "x" << std::endl;
    std::cout << "  Backward speedup (patches vs matrix): " << backward_speedup << "x" << std::endl;
}

int main( int argc, char *argv[] )
{
    (void)argc;
    (void)argv;

    std::cout << "================================================" << std::endl;
    std::cout << "   Convolution Diagnostics & Memory Analysis   " << std::endl;
    std::cout << "================================================" << std::endl;

    try
    {
        // Run diagnostic tests
        runDiagnosticTest( "basic_3x3", 1, 1, 3, 1, 0, 5, 5 );
        runDiagnosticTest( "strided", 1, 1, 3, 2, 1, 5, 5 );
        runDiagnosticTest( "multi_channel", 3, 16, 3, 1, 1, 8, 8 );
        runDiagnosticTest( "large_kernel", 1, 1, 5, 1, 2, 10, 10 );
        runDiagnosticTest( "depthwise", 8, 8, 3, 1, 1, 16, 16 );

        std::cout << "\n================================================" << std::endl;
        std::cout << "   Diagnostics Complete!                       " << std::endl;
        std::cout << "================================================" << std::endl;
        std::cout << "\nJSON files created for comparison with auto_LiRPA analysis" << std::endl;

        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}
