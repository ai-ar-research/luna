/*********************                                                        */
/*! \file TorchModelOnnxFactory.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief Path-based `TorchModel` constructors live in this translation
 ** unit so that they (and the ONNX / VNN-LIB parser code they pull in)
 ** are shipped in the `LUNAOnnxParser` library rather than `LUNACore`.
 ** Embedders that don't want the generated ONNX protobuf symbols in
 ** their binary link only `luna::core`; embedders that do want them
 ** opt in via `luna::onnx_parser`.
 **/

#include "TorchModel.h"
#include "LunaConfiguration.h"
#include "LunaError.h"
#include "OnnxToTorch.h"
#include "OutputConstraint.h"
#include "VnnLibInputParser.h"

#include <cstdio>

namespace NLR
{

TorchModel::TorchModel( const String &onnxPath )
    : _device( LunaConfiguration::getDevice() )
{
    log( Stringf( "[TorchModel] Constructor called with ONNX path: %s", onnxPath.ascii() ) );

    std::shared_ptr<TorchModel> parsedModel = OnnxToTorchParser::parse( onnxPath );

    if ( !parsedModel )
    {
        throw LunaError( LunaError::ONNX_PARSING_ERROR,
                         Stringf( "Failed to parse ONNX file: %s", onnxPath.ascii() ).ascii() );
    }

    _nodes = parsedModel->_nodes;
    _inputIndices = parsedModel->_inputIndices;
    _outputIndex = parsedModel->_outputIndex;
    _dependencies = parsedModel->_dependencies;
    _input_size = parsedModel->_input_size;
    _output_size = parsedModel->_output_size;
    _hasSpecificationMatrix = false;
    _hasORBranches = false;
    _hasFinalAnalysisBounds = false;
    _device = LunaConfiguration::getDevice();

    moveToDevice( _device );
    buildDependencyGraph();

    log( Stringf( "[TorchModel] Successfully loaded ONNX model with %u nodes", _nodes.size() ) );
}

TorchModel::TorchModel( const String &onnxPath, const String &vnnlibPath )
    : _device( LunaConfiguration::getDevice() )
{
    log( Stringf( "[TorchModel] Constructor called with ONNX path: %s and VNN-LIB path: %s",
                  onnxPath.ascii(), vnnlibPath.ascii() ) );

    std::shared_ptr<TorchModel> parsedModel = OnnxToTorchParser::parse( onnxPath );

    if ( !parsedModel )
    {
        throw LunaError( LunaError::ONNX_PARSING_ERROR,
                         Stringf( "Failed to parse ONNX file: %s", onnxPath.ascii() ).ascii() );
    }

    _nodes = parsedModel->_nodes;
    _inputIndices = parsedModel->_inputIndices;
    _outputIndex = parsedModel->_outputIndex;
    _dependencies = parsedModel->_dependencies;
    _input_size = parsedModel->_input_size;
    _output_size = parsedModel->_output_size;
    _hasSpecificationMatrix = false;
    _hasORBranches = false;
    _hasFinalAnalysisBounds = false;
    _device = LunaConfiguration::getDevice();

    moveToDevice( _device );
    buildDependencyGraph();

    log( Stringf( "[TorchModel] Successfully loaded ONNX model with %u nodes", _nodes.size() ) );

    log( Stringf( "[TorchModel] Parsing VNN-LIB file for input bounds: %s", vnnlibPath.ascii() ) );

    try
    {
        BoundedTensor<torch::Tensor> inputBounds =
            VnnLibInputParser::parseInputBounds( vnnlibPath, _input_size );

        setInputBounds( inputBounds );

        log( Stringf( "[TorchModel] Successfully set input bounds from VNN-LIB file" ) );

        try
        {
            log( Stringf( "[TorchModel] Parsing VNN-LIB file for output constraints" ) );
            OutputConstraintSet outputConstraints =
                VnnLibInputParser::parseOutputConstraints( vnnlibPath, _output_size );

            if ( outputConstraints.hasConstraints() )
            {
                log( Stringf( "[TorchModel] Found %u output constraints",
                              outputConstraints.getNumConstraints() ) );

                CMatrixResult cMatrixResult = outputConstraints.toCMatrix();
                torch::Tensor C = cMatrixResult.C;

                if ( LunaConfiguration::VERBOSE )
                {
                    printf( "[DEBUG TorchModel] Specification matrix created:\n" );
                    printf( "  Shape: [%lld, %lld, %lld]\n", (long long)C.size( 0 ),
                            (long long)C.size( 1 ), (long long)C.size( 2 ) );
                    printf( "  Number of constraints: %lld\n", (long long)C.size( 0 ) );
                    printf( "  Output dimension: %lld\n", (long long)C.size( 2 ) );
                    if ( C.numel() <= 50 )
                    {
                        printf( "  Full matrix:\n" );
                        for ( int i = 0; i < C.size( 0 ); ++i )
                        {
                            printf( "    Constraint %d: [", i );
                            for ( int j = 0; j < C.size( 2 ); ++j )
                            {
                                if ( j > 0 )
                                    printf( ", " );
                                printf( "%.3f", C[i][0][j].item<float>() );
                            }
                            printf( "]\n" );
                        }
                    }
                    else
                    {
                        printf( "  First constraint: [" );
                        for ( int j = 0; j < std::min( 10, (int)C.size( 2 ) ); ++j )
                        {
                            if ( j > 0 )
                                printf( ", " );
                            printf( "%.3f", C[0][0][j].item<float>() );
                        }
                        if ( C.size( 2 ) > 10 )
                            printf( ", ..." );
                        printf( "]\n" );
                    }
                    if ( cMatrixResult.thresholds.defined() )
                    {
                        printf( "  Thresholds: [" );
                        auto thresh_flat = cMatrixResult.thresholds.flatten();
                        for ( int i = 0; i < std::min( 10, (int)thresh_flat.numel() ); ++i )
                        {
                            if ( i > 0 )
                                printf( ", " );
                            printf( "%.6f", thresh_flat[i].item<float>() );
                        }
                        if ( thresh_flat.numel() > 10 )
                            printf( ", ..." );
                        printf( "]\n" );
                    }
                }

                setSpecificationFromConstraints( outputConstraints );

                log( Stringf(
                    "[TorchModel] Successfully set specification matrix with %d constraints",
                    (int)C.size( 0 ) ) );
            }
            else
            {
                log( Stringf( "[TorchModel] No output constraints found in VNN-LIB file" ) );
            }
        }
        catch ( const std::exception &e )
        {
            log( Stringf(
                "[TorchModel] Could not parse output constraints: %s (continuing without them)",
                e.what() ) );
        }
    }
    catch ( const LunaError &e )
    {
        throw;
    }
    catch ( const std::exception &e )
    {
        throw LunaError(
            LunaError::ONNX_PARSING_ERROR,
            Stringf( "Failed to parse VNN-LIB file %s: %s", vnnlibPath.ascii(), e.what() )
                .ascii() );
    }
}

} // namespace NLR
