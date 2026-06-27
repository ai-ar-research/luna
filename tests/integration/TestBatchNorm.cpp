#include "src/engine/TorchModel.h"
#include "src/engine/nodes/BoundedBatchNormNode.h"
#include "src/input_parsers/OnnxToTorch.h"
#include "src/common/Error.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>

// ONNX proto
#include "onnx.proto3.pb.h"

static std::vector<int64_t> readFirstInputShapeOrDefault( const std::string &onnxPath )
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
    if ( model.graph().input_size() < 1 )
    {
        return { 1, 1, 1, 1 };
    }

    const auto &input = model.graph().input( 0 );
    if ( !input.has_type() || !input.type().has_tensor_type() ||
         !input.type().tensor_type().has_shape() )
    {
        return { 1, 1, 1, 1 };
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
        dims = { 1, 1, 1, 1 };
    return dims;
}

int main( int argc, char *argv[] )
{
    try
    {
        std::string onnxFilePath;
        if ( argc >= 2 )
        {
            onnxFilePath = argv[1];
        }
        else
        {
            // Try a few common working directories (repo root vs luna/build).
            const char *candidates[] = {
                "../resources/onnx/layer-zoo/batchnorm.onnx",  // when cwd is luna/build
                "resources/onnx/layer-zoo/batchnorm.onnx",     // when cwd is luna/
                "luna/resources/onnx/layer-zoo/batchnorm.onnx" // when cwd is repo root
            };
            for ( const char *p : candidates )
            {
                std::ifstream fin( p, std::ios::binary );
                if ( fin.is_open() )
                {
                    onnxFilePath = p;
                    break;
                }
            }
            if ( onnxFilePath.empty() )
            {
                onnxFilePath = candidates[0];
            }
        }

        std::cout << "Parsing ONNX file: " << onnxFilePath << "\n\n";

        NLR::TorchModel model( String( onnxFilePath.c_str() ) );

        auto bnNodes = model.getNodesByType( NLR::NodeType::BATCHNORM );
        if ( bnNodes.empty() )
        {
            std::cerr << "[FAIL] No BATCHNORM nodes found in parsed model.\n";
            return 1;
        }
        unsigned bnIdx = bnNodes[0];
        auto bnBase = model.getNode( bnIdx );
        auto bn = std::dynamic_pointer_cast<NLR::BoundedBatchNormNode>( bnBase );
        if ( !bn )
        {
            std::cerr << "[FAIL] Node type is BATCHNORM but dynamic cast failed.\n";
            return 1;
        }

        auto deps = model.getDependencies( bnIdx );
        if ( deps.empty() )
        {
            std::cerr << "[FAIL] BATCHNORM node has no dependency input.\n";
            return 1;
        }
        unsigned xIdx = deps[0];

        // Build an input tensor with the ONNX-declared shape (fallbacks to all-ones if unknown).
        auto inShape = readFirstInputShapeOrDefault( onnxFilePath );
        torch::Tensor x_in =
            torch::randn( inShape, torch::TensorOptions().dtype( torch::kFloat32 ) );

        // Forward pass and extract BN input/output activations.
        auto acts = model.forwardAndStoreActivations( x_in );
        torch::Tensor x = acts[xIdx];
        torch::Tensor y = acts[bnIdx];

        // Validate affine form: y == w*x + b (with per-channel w,b)
        torch::Tensor w = bn->getTmpWeight( x );
        torch::Tensor b = bn->getTmpBias( x );

        // Channel broadcast (N,C,...) where C is dim 1.
        std::vector<int64_t> view( x.dim(), 1 );
        if ( x.dim() >= 2 )
            view[1] = w.numel();
        torch::Tensor wv = ( x.dim() >= 2 ) ? w.view( view ) : w;
        torch::Tensor bv = ( x.dim() >= 2 ) ? b.view( view ) : b;

        torch::Tensor y_expected = wv * x + bv;
        auto max_err = ( y - y_expected ).abs().max().item<float>();
        std::cout << "Forward max |y - (w*x+b)| = " << max_err << "\n";
        if ( max_err > 1e-4f )
        {
            std::cerr << "[FAIL] Forward check failed.\n";
            return 1;
        }

        // IBP check: exact interval (lower==upper==x) should match forward exactly.
        Vector<BoundedTensor<torch::Tensor>> ibpIn;
        ibpIn.append( BoundedTensor<torch::Tensor>( x, x ) );
        auto y_ibp = bn->computeIntervalBoundPropagation( ibpIn );
        auto ibp_err_l = ( y_ibp.lower() - y_expected ).abs().max().item<float>();
        auto ibp_err_u = ( y_ibp.upper() - y_expected ).abs().max().item<float>();
        std::cout << "IBP exact interval max err lower=" << ibp_err_l << " upper=" << ibp_err_u
                  << "\n";
        if ( ibp_err_l > 1e-4f || ibp_err_u > 1e-4f )
        {
            std::cerr << "[FAIL] IBP check failed.\n";
            return 1;
        }

        // Backward bound check (Tensor mode): last_A is ones like y -> next_A should be ones*w
        NLR::BoundA lastA( torch::ones_like( y ) );
        Vector<Pair<NLR::BoundA, NLR::BoundA>> outAs;
        torch::Tensor lbias, ubias;
        Vector<BoundedTensor<torch::Tensor>> inputBounds;
        inputBounds.append( BoundedTensor<torch::Tensor>( x, x ) );
        bn->boundBackward( lastA, lastA, inputBounds, outAs, lbias, ubias );

        if ( outAs.empty() || !outAs[0].first().defined() || !outAs[0].first().isTensor() )
        {
            std::cerr << "[FAIL] boundBackward did not produce tensor A.\n";
            return 1;
        }
        torch::Tensor nextA = outAs[0].first().asTensor();
        torch::Tensor nextA_expected = torch::ones_like( y ) * wv;
        auto bw_err = ( nextA - nextA_expected ).abs().max().item<float>();
        std::cout << "Backward max |nextA - (ones*w)| = " << bw_err << "\n";
        if ( bw_err > 1e-4f )
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
