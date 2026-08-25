#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "src/configuration/LunaConfiguration.h"
#include "src/common/BoundedTensor.h"
#include "src/engine/nodes/BoundedInputNode.h"
#include "src/engine/nodes/BoundedConvNode.h"
#include "src/engine/nodes/BoundedBatchNormNode.h"
#include "src/engine/nodes/BoundedReLUNode.h"
#include "src/engine/nodes/BoundedAddNode.h"
#
#include <torch/torch.h>
#include <iostream>
#include <memory>
#
using namespace NLR;
#
static void assertCenterContained( TorchModel &model,
                                   const BoundedTensor<torch::Tensor> &inputBounds,
                                   const BoundedTensor<torch::Tensor> &outBounds )
{
    torch::NoGradGuard no_grad;
    torch::Tensor center = ( inputBounds.lower() + inputBounds.upper() ) / 2.0;
    // Match TorchModel's reshape heuristic for common image shapes.
    if ( center.dim() == 1 && center.numel() == 3072 )
    {
        center = center.view( { 1, 3, 32, 32 } );
    }
    auto acts = model.forwardAndStoreActivations( center );
    torch::Tensor y = acts[model.getOutputIndex()].flatten().to( torch::kFloat32 );
    torch::Tensor lb = outBounds.lower().flatten().to( torch::kFloat32 );
    torch::Tensor ub = outBounds.upper().flatten().to( torch::kFloat32 );
    int64_t n = std::min<int64_t>( y.numel(), std::min<int64_t>( lb.numel(), ub.numel() ) );
    int64_t bad = 0;
    for ( int64_t i = 0; i < n; ++i )
    {
        float yi = y[i].item<float>();
        float lbi = lb[i].item<float>();
        float ubi = ub[i].item<float>();
        if ( !( lbi <= yi && yi <= ubi ) )
            bad++;
    }
    if ( bad > 0 )
    {
        std::ostringstream oss;
        oss << "[FAIL] Center containment check failed: " << ( n - bad ) << "/" << n
            << " contained";
        throw std::runtime_error( oss.str() );
    }
}
#
int main( int argc, char **argv )
{
    (void)argc;
    (void)argv;
#
    try
    {
        // Small residual block: x -> conv1 -> bn1 -> relu -> conv2 -> bn2 ; y = bn2 + x
        // This creates a join point in the backward graph (x feeds two dependents), which must
        // be handled correctly by the CROWN backward scheduler.
#
        const int64_t N = 1;
        const int64_t C = 3;
        const int64_t H = 32;
        const int64_t W = 32;
        // Input node
        auto in = std::make_shared<BoundedInputNode>(
            /*inputIndex=*/0, /*inputSize=*/(unsigned)( C * H * W ), "input" );
#
        // Conv1: 3->3, stride=1, padding=1
        torch::nn::Conv2d conv1( torch::nn::Conv2dOptions( C, C, /*kernel_size=*/3 )
                                     .stride( 1 )
                                     .padding( 1 )
                                     .bias( false ) );
        conv1->weight = torch::randn_like( conv1->weight ) * 0.05;
        auto nConv1 = std::make_shared<BoundedConvNode>( conv1, ConvMode::MATRIX, "conv1" );
#
        // BN1 params
        torch::Tensor scale1 = torch::ones( { C }, torch::kFloat32 );
        torch::Tensor bias1 = torch::zeros( { C }, torch::kFloat32 );
        torch::Tensor mean1 = torch::zeros( { C }, torch::kFloat32 );
        torch::Tensor var1 = torch::ones( { C }, torch::kFloat32 );
        auto bn1 = std::make_shared<BoundedBatchNormNode>( scale1, bias1, mean1, var1,
                                                           /*eps=*/1e-5f, "bn1" );
#
        auto relu = std::make_shared<BoundedReLUNode>( torch::nn::ReLU(), "relu" );
#
        // Conv2: 3->3, stride=1, padding=1
        torch::nn::Conv2d conv2( torch::nn::Conv2dOptions( C, C, /*kernel_size=*/3 )
                                     .stride( 1 )
                                     .padding( 1 )
                                     .bias( false ) );
        conv2->weight = torch::randn_like( conv2->weight ) * 0.05;
        auto nConv2 = std::make_shared<BoundedConvNode>( conv2, ConvMode::MATRIX, "conv2" );
#
        // BN2 params
        torch::Tensor scale2 = torch::ones( { C }, torch::kFloat32 );
        torch::Tensor bias2 = torch::zeros( { C }, torch::kFloat32 );
        torch::Tensor mean2 = torch::zeros( { C }, torch::kFloat32 );
        torch::Tensor var2 = torch::ones( { C }, torch::kFloat32 );
        auto bn2 = std::make_shared<BoundedBatchNormNode>( scale2, bias2, mean2, var2,
                                                           /*eps=*/1e-5f, "bn2" );
#
        auto add = std::make_shared<BoundedAddNode>();
        add->setNodeName( "add" );
#
        // Nodes vector
        Vector<std::shared_ptr<BoundedTorchNode>> nodes;
        nodes.append( in );     // 0
        nodes.append( nConv1 ); // 1
        nodes.append( bn1 );    // 2
        nodes.append( relu );   // 3
        nodes.append( nConv2 ); // 4
        nodes.append( bn2 );    // 5
        nodes.append( add );    // 6 (output)
#
        // Dependencies (nodeIndex -> input node indices)
        Map<unsigned, Vector<unsigned>> deps;
        deps[0] = Vector<unsigned>(); // input
        deps[1] = Vector<unsigned>( { 0 } );
        deps[2] = Vector<unsigned>( { 1 } );
        deps[3] = Vector<unsigned>( { 2 } );
        deps[4] = Vector<unsigned>( { 3 } );
        deps[5] = Vector<unsigned>( { 4 } );
        deps[6] = Vector<unsigned>( { 5, 0 } ); // residual add: bn2 + input
#
        Vector<unsigned> inputIndices;
        inputIndices.append( 0 );
        unsigned outputIndex = 6;
#
        TorchModel model( nodes, inputIndices, outputIndex, deps );
#
        // Center input in conv layout.
        torch::Tensor center =
            torch::randn( { N, C, H, W }, torch::TensorOptions().dtype( torch::kFloat32 ) );
        float eps = 1e-3f;
        // Preserve the declared image layout so the residual merge exercises mixed Tensor/Patches A
        // accumulation.
        torch::Tensor lb = center - eps;
        torch::Tensor ub = center + eps;
        BoundedTensor<torch::Tensor> inputBounds( lb, ub );
        model.setInputBounds( inputBounds );

        // Run a single forward pass to populate per-node shapes and explicitly set node sizes.
        // Some nodes (e.g., Add) don't infer _output_size in forward(), but CROWN needs outputSize
        // != 0.
        {
            torch::NoGradGuard no_grad;
            auto acts = model.forwardAndStoreActivations( center );
            for ( unsigned i = 0; i < nodes.size(); ++i )
            {
                if ( !nodes[i] )
                    continue;
                nodes[i]->setNodeIndex( i );

                // Output size per batch from activation.
                if ( acts.exists( i ) )
                {
                    torch::Tensor t = acts[i];
                    int64_t b = ( t.defined() && t.dim() > 0 ) ? t.size( 0 ) : 1;
                    if ( b <= 0 )
                        b = 1;
                    unsigned outSz = (unsigned)( t.numel() / b );
                    nodes[i]->setOutputSize( outSz );
                }

                // Input size per batch from first dependency activation if available.
                if ( deps.exists( i ) && deps[i].size() > 0 )
                {
                    unsigned dep0 = deps[i][0];
                    if ( acts.exists( dep0 ) )
                    {
                        torch::Tensor tin = acts[dep0];
                        int64_t b = ( tin.defined() && tin.dim() > 0 ) ? tin.size( 0 ) : 1;
                        if ( b <= 0 )
                            b = 1;
                        unsigned inSz = (unsigned)( tin.numel() / b );
                        nodes[i]->setInputSize( inSz );
                    }
                }
            }
        }
#
        // CROWN bounds must be sound: center output contained.
        auto out = model.compute_bounds( inputBounds, nullptr,
                                         LunaConfiguration::AnalysisMethod::CROWN, true, true );
        assertCenterContained( model, inputBounds, out );
#
        std::cout << "Passed\n";
        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
