#include "src/engine/TorchModel.h"
#include "src/engine/nodes/BoundedAddNode.h"
#include "src/common/Error.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>

// ONNX proto
#include "onnx.proto3.pb.h"

static std::vector<int64_t> readInputShapeOrDefault( const std::string &onnxPath, int inputIdx,
                                                     const std::vector<int64_t> &fallback = { 1,
                                                                                              4 } )
{
    luna_onnx::ModelProto model;
    std::ifstream fin( onnxPath, std::ios::binary );
    if ( !fin.is_open() )
    {
        throw std::runtime_error( "Failed to open ONNX file: " + onnxPath );
    }
    if ( !model.ParseFromIstream( &fin ) )
    {
        throw std::runtime_error( "Failed to parse ONNX model: " + onnxPath );
    }
    if ( model.graph().input_size() <= inputIdx )
    {
        return fallback;
    }

    const auto &input = model.graph().input( inputIdx );
    if ( !input.has_type() || !input.type().has_tensor_type() ||
         !input.type().tensor_type().has_shape() )
    {
        return fallback;
    }

    const auto &shape = input.type().tensor_type().shape();
    std::vector<int64_t> dims;
    dims.reserve( shape.dim_size() );
    for ( int i = 0; i < shape.dim_size(); ++i )
    {
        const auto &d = shape.dim( i );
        int64_t v = 1;
        if ( d.has_dim_value() )
            v = static_cast<int64_t>( d.dim_value() );
        if ( v <= 0 )
            v = 1;
        dims.push_back( v );
    }
    if ( dims.empty() )
        dims = fallback;
    return dims;
}

static torch::Tensor expectedBroadcastBackwardElementwise( const torch::Tensor &A,
                                                           const torch::Tensor &operand )
{
    // Compute the broadcast-backward reduction for elementwise A (same shape as output).
    // A shape: [B, ...] (output shape)
    // operand shape: [B, ...] (possibly with fewer dims / singleton broadcast dims)
    if ( !A.defined() )
        return A;
    if ( !operand.defined() )
        return A;
    if ( A.dim() < 1 || operand.dim() < 1 )
        return A;

    torch::Tensor out = A;
    int64_t y_payload = out.dim() - 1;     // excluding batch
    int64_t x_payload = operand.dim() - 1; // excluding batch

    // Sum extra leading payload dims in y (after batch) that x doesn't have.
    if ( y_payload > x_payload )
    {
        int64_t extra = y_payload - x_payload;
        std::vector<int64_t> sum_dims;
        sum_dims.reserve( (size_t)extra );
        for ( int64_t i = 0; i < extra; ++i )
        {
            sum_dims.push_back( 1 + i ); // leading payload dims
        }
        out = out.sum( sum_dims );
    }

    // Now ranks should match: out.dim() == operand.dim()
    // Sum broadcasted singleton dims with keepdim.
    std::vector<int64_t> keep_dims;
    int64_t common_dim = std::min( out.dim(), operand.dim() );
    for ( int64_t d = 1; d < common_dim; ++d )
    {
        if ( operand.size( d ) == 1 && out.size( d ) != 1 )
        {
            keep_dims.push_back( d );
        }
    }
    if ( !keep_dims.empty() )
    {
        out = out.sum( keep_dims, /*keepdim=*/true );
    }
    return out;
}

int main( int argc, char *argv[] )
{
    (void)argc;
    (void)argv;

    try
    {
        std::string onnxFilePath = std::string( RESOURCES_DIR ) + "/onnx/multiInput_add.onnx";

        std::cout << "Parsing ONNX file: " << onnxFilePath << "\n\n";

        NLR::TorchModel model( String( onnxFilePath.c_str() ) );

        auto addNodes = model.getNodesByType( NLR::NodeType::ADD );
        if ( addNodes.empty() )
        {
            std::cerr << "[FAIL] No ADD nodes found in parsed model.\n";
            return 1;
        }
        unsigned addIdx = addNodes[0];

        auto addBase = model.getNode( addIdx );
        auto add = std::dynamic_pointer_cast<NLR::BoundedAddNode>( addBase );
        if ( !add )
        {
            std::cerr << "[FAIL] Node type is ADD but dynamic cast failed.\n";
            return 1;
        }

        auto deps = model.getDependencies( addIdx );
        if ( deps.size() != 2 )
        {
            std::cerr << "[FAIL] ADD node expected 2 dependencies, got " << deps.size() << "\n";
            return 1;
        }
        unsigned x0Idx = deps[0];
        unsigned x1Idx = deps[1];

        const auto &inputIdxs = model.getInputIndices();
        if ( inputIdxs.size() < 2 )
        {
            std::cerr << "[FAIL] Model expected at least 2 input nodes, got " << inputIdxs.size()
                      << "\n";
            return 1;
        }

        // Build input tensors from ONNX-declared shapes (with a robust fallback to model input
        // sizes).
        unsigned in0Size = model.getNode( inputIdxs[0] )->getOutputSize();
        unsigned in1Size = model.getNode( inputIdxs[1] )->getOutputSize();

        auto in0Shape = readInputShapeOrDefault( onnxFilePath, 0, { 1, (int64_t)in0Size } );
        auto in1Shape = readInputShapeOrDefault( onnxFilePath, 1, { 1, (int64_t)in1Size } );

        // Heuristic fixup: if ONNX provides a suspicious [features, 1] shape, transpose to [1,
        // features].
        if ( in0Shape.size() == 2 && in0Shape[0] != 1 && in0Shape[1] == 1 )
        {
            in0Shape = { 1, in0Shape[0] };
        }
        if ( in1Shape.size() == 2 && in1Shape[0] != 1 && in1Shape[1] == 1 )
        {
            in1Shape = { 1, in1Shape[0] };
        }

        torch::Tensor x0_in =
            torch::randn( in0Shape, torch::TensorOptions().dtype( torch::kFloat32 ) );
        torch::Tensor x1_in =
            torch::randn( in1Shape, torch::TensorOptions().dtype( torch::kFloat32 ) );

        Map<unsigned, torch::Tensor> inputs;
        inputs[inputIdxs[0]] = x0_in;
        inputs[inputIdxs[1]] = x1_in;

        auto acts = model.forwardAndStoreActivations( inputs );
        if ( !acts.exists( x0Idx ) || !acts.exists( x1Idx ) || !acts.exists( addIdx ) )
        {
            std::cerr << "[FAIL] Missing activations for Add dependencies/output.\n";
            return 1;
        }

        torch::Tensor x0 = acts[x0Idx];
        torch::Tensor x1 = acts[x1Idx];
        torch::Tensor y = acts[addIdx];

        torch::Tensor y_expected = x0 + x1;
        auto max_err = ( y - y_expected ).abs().max().item<float>();
        std::cout << "Forward max |y - (x0+x1)| = " << max_err << "\n";
        if ( max_err > 1e-5f )
        {
            std::cerr << "[FAIL] Forward check failed.\n";
            return 1;
        }

        // IBP exact interval should match forward exactly.
        Vector<BoundedTensor<torch::Tensor>> ibpIn;
        ibpIn.append( BoundedTensor<torch::Tensor>( x0, x0 ) );
        ibpIn.append( BoundedTensor<torch::Tensor>( x1, x1 ) );
        auto y_ibp = add->computeIntervalBoundPropagation( ibpIn );
        auto ibp_err_l = ( y_ibp.lower() - y_expected ).abs().max().item<float>();
        auto ibp_err_u = ( y_ibp.upper() - y_expected ).abs().max().item<float>();
        std::cout << "IBP exact interval max err lower=" << ibp_err_l << " upper=" << ibp_err_u
                  << "\n";
        if ( ibp_err_l > 1e-5f || ibp_err_u > 1e-5f )
        {
            std::cerr << "[FAIL] IBP check failed.\n";
            return 1;
        }

        // Backward check: ones_like(y) should propagate to each input, reduced for broadcasting if
        // needed.
        NLR::BoundA lastA( torch::ones_like( y ) );
        Vector<Pair<NLR::BoundA, NLR::BoundA>> outAs;
        torch::Tensor lbias, ubias;
        add->boundBackward( lastA, lastA, ibpIn, outAs, lbias, ubias );

        if ( outAs.size() != 2 )
        {
            std::cerr << "[FAIL] boundBackward expected 2 outputs, got " << outAs.size() << "\n";
            return 1;
        }
        if ( !outAs[0].first().defined() || !outAs[0].first().isTensor() )
        {
            std::cerr << "[FAIL] boundBackward did not produce tensor A for input 0.\n";
            return 1;
        }
        if ( !outAs[1].first().defined() || !outAs[1].first().isTensor() )
        {
            std::cerr << "[FAIL] boundBackward did not produce tensor A for input 1.\n";
            return 1;
        }

        torch::Tensor nextA0 = outAs[0].first().asTensor();
        torch::Tensor nextA1 = outAs[1].first().asTensor();

        torch::Tensor expA0 = expectedBroadcastBackwardElementwise( torch::ones_like( y ), x0 );
        torch::Tensor expA1 = expectedBroadcastBackwardElementwise( torch::ones_like( y ), x1 );

        auto bw_err0 = ( nextA0 - expA0 ).abs().max().item<float>();
        auto bw_err1 = ( nextA1 - expA1 ).abs().max().item<float>();
        std::cout << "Backward max |nextA0 - expected| = " << bw_err0 << "\n";
        std::cout << "Backward max |nextA1 - expected| = " << bw_err1 << "\n";
        if ( bw_err0 > 1e-5f || bw_err1 > 1e-5f )
        {
            std::cerr << "[FAIL] backward A check failed.\n";
            return 1;
        }

        std::cout << "\nPassed\n";
        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch ( const Error &e )
    {
        std::cerr << "Error: " << e.getErrorClass() << " code=" << e.getCode();
        if ( e.getUserMessage() )
            std::cerr << " msg=" << e.getUserMessage();
        std::cerr << "\n";
        return 1;
    }
}
