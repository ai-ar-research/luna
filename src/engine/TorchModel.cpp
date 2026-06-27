/*********************                                                        */
/*! \file TorchModel.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "TorchModel.h"
#include "CROWNAnalysis.h"
#include "AlphaCROWNAnalysis.h"
#include "LunaError.h"
#include "LunaConfiguration.h"
#include "OutputConstraint.h"
#include <iostream>

namespace NLR
{

TorchModel::TorchModel( const Vector<std::shared_ptr<BoundedTorchNode>> &nodes,
                        const Vector<unsigned> &inputIndices, unsigned outputIndex,
                        const Map<unsigned, Vector<unsigned>> &dependencies )
    : _nodes( nodes )
    , _inputIndices( inputIndices )
    , _outputIndex( outputIndex )
    , _dependencies( dependencies )
    , _input_size( 0 )
    , _output_size( 0 )
    , _hasSpecificationMatrix( false )
    , _hasORBranches( false )
    , _hasFinalAnalysisBounds( false )
    , _device( LunaConfiguration::getDevice() )
{

    log( Stringf( "[TorchModel] Constructor called with %u nodes", nodes.size() ) );

    if ( !inputIndices.empty() && inputIndices[0] < nodes.size() )
    {
        auto inputNode = nodes[inputIndices[0]];
        if ( inputNode )
        {
            _input_size = inputNode->getOutputSize();
            log( Stringf( "[TorchModel] Set input size to %u from input node %u", _input_size,
                          inputIndices[0] ) );
        }
    }

    if ( outputIndex < nodes.size() )
    {
        auto outputNode = nodes[outputIndex];
        if ( outputNode )
        {
            _output_size = outputNode->getOutputSize();
            log( Stringf( "[TorchModel] Set output size to %u from output node %u", _output_size,
                          outputIndex ) );
        }
    }

    moveToDevice( _device );
    buildDependencyGraph();
}

// NOTE: the two path-based ctors (TorchModel(onnxPath) and
// TorchModel(onnxPath, vnnlibPath)) live in TorchModelOnnxFactory.cpp,
// shipped via LUNAOnnxParser. Embedders that don't want the generated
// ONNX protobuf symbols in their binary link only luna::core.

void TorchModel::log( const String &message ) const
{
    (void)message;
    if ( LunaConfiguration::NETWORK_LEVEL_REASONER_LOGGING )
    {
    }
}

void TorchModel::buildDependencyGraph()
{
    _dependents.clear();
    _degreeOut.clear();
    _degreeIn.clear();
    _processed.clear();

    for ( const auto &pair : _dependencies )
    {
        unsigned nodeIndex = pair.first;
        _dependents[nodeIndex] = Vector<unsigned>();
        _degreeOut[nodeIndex] = 0;
        _degreeIn[nodeIndex] = 0;
        _processed[nodeIndex] = false;
    }

    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( !_dependents.exists( i ) )
        {
            _dependents[i] = Vector<unsigned>();
            _degreeOut[i] = 0;
            _degreeIn[i] = 0;
            _processed[i] = false;
        }
    }

    buildDependents();
    computeDegrees();
}

void TorchModel::buildDependents()
{
    for ( const auto &pair : _dependencies )
    {
        unsigned nodeIndex = pair.first;

        for ( unsigned inputIndex : _dependencies[nodeIndex] )
        {
            _dependents[inputIndex].append( nodeIndex );
        }
    }
}

void TorchModel::computeDegrees()
{
    for ( const auto &pair : _dependents )
    {
        unsigned nodeIndex = pair.first;
        _degreeOut[nodeIndex] = _dependents[nodeIndex].size();
    }

    for ( const auto &pair : _dependencies )
    {
        unsigned nodeIndex = pair.first;
        _degreeIn[nodeIndex] = _dependencies[nodeIndex].size();
    }
}

void TorchModel::resetProcessingState()
{
    for ( auto &pair : _processed )
    {
        pair.second = false;
    }
}

Vector<unsigned> TorchModel::topologicalSort() const
{
    Vector<unsigned> sortedOrder;
    Queue<unsigned> queue;

    Map<unsigned, unsigned> degreeIn = _degreeIn;

    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( degreeIn.exists( i ) && degreeIn[i] == 0 )
        {
            queue.push( i );
        }
    }

    while ( !queue.empty() )
    {
        unsigned current = queue.peak();
        queue.pop();
        sortedOrder.append( current );

        if ( _dependents.exists( current ) )
        {
            for ( unsigned dependent : _dependents[current] )
            {
                if ( degreeIn.exists( dependent ) )
                {
                    degreeIn[dependent]--;
                    if ( degreeIn[dependent] == 0 )
                    {
                        queue.push( dependent );
                    }
                }
            }
        }
    }

    return sortedOrder;
}

Vector<unsigned> TorchModel::getRoots() const
{
    Vector<unsigned> roots;
    for ( const auto &pair : _degreeIn )
    {
        if ( pair.second == 0 )
        {
            roots.append( pair.first );
        }
    }
    return roots;
}

Vector<unsigned> TorchModel::getLeaves() const
{
    Vector<unsigned> leaves;
    for ( const auto &pair : _degreeOut )
    {
        if ( pair.second == 0 )
        {
            leaves.append( pair.first );
        }
    }
    return leaves;
}

Vector<unsigned> TorchModel::getDependents( unsigned nodeIndex ) const
{
    if ( _dependents.exists( nodeIndex ) )
    {
        return _dependents[nodeIndex];
    }
    return Vector<unsigned>();
}

Vector<unsigned> TorchModel::getDependencies( unsigned nodeIndex ) const
{
    if ( _dependencies.exists( nodeIndex ) )
    {
        return _dependencies[nodeIndex];
    }
    return Vector<unsigned>();
}

unsigned TorchModel::getDegreeOut( unsigned nodeIndex ) const
{
    if ( _degreeOut.exists( nodeIndex ) )
    {
        return _degreeOut[nodeIndex];
    }
    return 0;
}

unsigned TorchModel::getDegreeIn( unsigned nodeIndex ) const
{
    if ( _degreeIn.exists( nodeIndex ) )
    {
        return _degreeIn[nodeIndex];
    }
    return 0;
}

bool TorchModel::isProcessed( unsigned nodeIndex ) const
{
    if ( _processed.exists( nodeIndex ) )
    {
        return _processed[nodeIndex];
    }
    return false;
}

void TorchModel::markProcessed( unsigned nodeIndex )
{
    if ( _processed.exists( nodeIndex ) )
    {
        _processed[nodeIndex] = true;
    }
}

std::shared_ptr<BoundedTorchNode> TorchModel::getNode( unsigned index ) const
{
    if ( index < _nodes.size() )
    {
        return _nodes[index];
    }
    return nullptr;
}

Vector<unsigned> TorchModel::getAllNodeIndices() const
{
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        indices.append( i );
    }
    return indices;
}

Vector<unsigned> TorchModel::getNodesByType( NodeType type ) const
{
    Vector<unsigned> indices;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( _nodes[i]->getNodeType() == type )
        {
            indices.append( i );
        }
    }
    return indices;
}

void TorchModel::moveToDevice( const torch::Device &device )
{
    _device = device;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( _nodes[i] )
        {
            _nodes[i]->moveToDevice( device );
        }
    }
    if ( _inputBounds.lower().defined() || _inputBounds.upper().defined() )
    {
        _inputBounds = BoundedTensor<torch::Tensor>( _inputBounds.lower().to( device ),
                                                     _inputBounds.upper().to( device ) );
    }
    if ( _specificationMatrix.defined() )
    {
        _specificationMatrix = _specificationMatrix.to( device );
    }
    if ( _specificationThresholds.defined() )
    {
        _specificationThresholds = _specificationThresholds.to( device );
    }
    if ( _finalAnalysisBounds.lower().defined() || _finalAnalysisBounds.upper().defined() )
    {
        _finalAnalysisBounds = BoundedTensor<torch::Tensor>(
            _finalAnalysisBounds.lower().to( device ), _finalAnalysisBounds.upper().to( device ) );
    }
}

torch::Tensor TorchModel::forward( unsigned nodeIndex, Map<unsigned, torch::Tensor> &activations,
                                   const Map<unsigned, torch::Tensor> &inputs )
{
    if ( activations.exists( nodeIndex ) )
    {
        return activations[nodeIndex];
    }

    if ( nodeIndex >= _nodes.size() )
    {
        throw LunaError(
            LunaError::INVALID_MODEL_STRUCTURE,
            ( String( "Node index not found: " ) + std::to_string( nodeIndex ) ).ascii() );
    }

    auto &node = _nodes[nodeIndex];
    NodeType nodeType = node->getNodeType();

    switch ( nodeType )
    {
    case NodeType::INPUT:
    {
        unsigned inputIndex = nodeIndex;
        if ( !inputs.exists( inputIndex ) )
        {
            throw LunaError(
                LunaError::INVALID_MODEL_STRUCTURE,
                ( String( "Input index not found: " ) + std::to_string( inputIndex ) ).ascii() );
        }
        activations[nodeIndex] = inputs[inputIndex];
        return inputs[inputIndex];
    }

    case NodeType::CONSTANT:
    {
        torch::Tensor result = node->forward( torch::Tensor() );
        activations[nodeIndex] = result;
        return result;
    }

    case NodeType::LINEAR:
    case NodeType::RELU:
    case NodeType::RESHAPE:
    case NodeType::FLATTEN:
    case NodeType::IDENTITY:
    case NodeType::SUB:
    case NodeType::ADD:
    case NodeType::CONV:
    case NodeType::BATCHNORM:
    case NodeType::SIGMOID:
    case NodeType::CONCAT:
    case NodeType::CONVTRANSPOSE:
    case NodeType::SLICE:
    {
        if ( !_dependencies.exists( nodeIndex ) )
        {
            throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                             ( String( "No dependencies found for node at index: " ) +
                               std::to_string( nodeIndex ) )
                                 .ascii() );
        }

        Vector<unsigned> deps = _dependencies[nodeIndex];

        std::vector<torch::Tensor> inputTensors;
        for ( unsigned dep : deps )
        {
            inputTensors.push_back( forward( dep, activations, inputs ) );
        }

        if ( inputTensors.empty() )
        {
            throw LunaError(
                LunaError::INVALID_MODEL_STRUCTURE,
                ( String( "No input tensors for node at index: " ) + std::to_string( nodeIndex ) )
                    .ascii() );
        }

        torch::Tensor result;
        if ( inputTensors.size() == 1 )
        {
            result = node->forward( inputTensors[0] );
        }
        else
        {
            result = node->forward( inputTensors );
        }

        activations[nodeIndex] = result;
        return result;
    }
    }

    throw LunaError(
        LunaError::INVALID_MODEL_STRUCTURE,
        ( String( "Unknown node type for node index: " ) + std::to_string( nodeIndex ) ).ascii() );
}

Map<unsigned, torch::Tensor> TorchModel::forwardAndStoreActivations( const torch::Tensor &input )
{
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> inputs_map;
    torch::Tensor deviceInput = input.to( _device );

    for ( unsigned inputIndex : _inputIndices )
    {
        inputs_map[inputIndex] = deviceInput;
    }

    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        forward( i, activations, inputs_map );
    }

    return activations;
}

Map<unsigned, torch::Tensor>
TorchModel::forwardAndStoreActivations( const Map<unsigned, torch::Tensor> &inputs )
{
    Map<unsigned, torch::Tensor> activations;
    Map<unsigned, torch::Tensor> deviceInputs;

    for ( auto it = inputs.begin(); it != inputs.end(); ++it )
    {
        deviceInputs[it->first] = it->second.to( _device );
    }

    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        forward( i, activations, deviceInputs );
    }

    return activations;
}

void TorchModel::setInputBounds( const BoundedTensor<torch::Tensor> &inputBounds )
{
    log( Stringf( "[TorchModel] Setting input bounds" ) );

    // detach to avoid retaining gradient history
    _inputBounds = BoundedTensor<torch::Tensor>( inputBounds.lower().detach().to( _device ),
                                                 inputBounds.upper().detach().to( _device ) );

    log( Stringf( "[TorchModel] Input bounds set with shape: %s",
                  inputBounds.lower().sizes().vec().data() ) );
}

void TorchModel::setConcreteBounds( unsigned nodeIndex,
                                    const BoundedTensor<torch::Tensor> &concreteBounds )
{
    validateNodeIndex( nodeIndex );

    _concreteBounds[nodeIndex] = concreteBounds;
    log( Stringf( "[TorchModel] Set concrete bounds for node %u", nodeIndex ) );
}

void TorchModel::clearConcreteBounds()
{
    _concreteBounds.clear();
    log( Stringf( "[TorchModel] Cleared all concrete bounds" ) );
}

BoundedTensor<torch::Tensor> TorchModel::getConcreteBounds( unsigned nodeIndex ) const
{
    validateNodeIndex( nodeIndex );

    if ( !hasConcreteBounds( nodeIndex ) )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                         Stringf( "Concrete bounds not computed for node %u", nodeIndex ).ascii() );
    }

    return _concreteBounds[nodeIndex];
}

bool TorchModel::hasConcreteBounds( unsigned nodeIndex ) const
{
    validateNodeIndex( nodeIndex );
    return _concreteBounds.exists( nodeIndex );
}

BoundedTensor<torch::Tensor> TorchModel::getInputBounds() const
{
    if ( !hasInputBounds() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No input bounds set" );
    }
    return _inputBounds;
}

bool TorchModel::hasInputBounds() const
{
    return _inputBounds.lower().defined() && _inputBounds.upper().defined();
}

torch::Tensor TorchModel::getInputLowerBounds() const
{
    if ( !hasInputBounds() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No input bounds set" );
    }
    return _inputBounds.lower();
}

torch::Tensor TorchModel::getInputUpperBounds() const
{
    if ( !hasInputBounds() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No input bounds set" );
    }
    return _inputBounds.upper();
}

void TorchModel::setSpecificationMatrix( const torch::Tensor &specMatrix )
{
    log( Stringf( "[TorchModel] Setting specification matrix with shape [%ld, %ld, %ld]",
                  specMatrix.size( 0 ), specMatrix.size( 1 ), specMatrix.size( 2 ) ) );
    _specificationMatrix = specMatrix.to( _device );
    _hasSpecificationMatrix = true;
    _specificationThresholds = torch::Tensor();
    _specificationBranchMapping.clear();
    _specificationBranchSizes.clear();
    _hasORBranches = false;
}

void TorchModel::setSpecificationFromConstraints( const OutputConstraintSet &constraints )
{
    log( "[TorchModel] Setting specification from OutputConstraintSet" );

    if ( !constraints.hasConstraints() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "OutputConstraintSet is empty" );
    }

    CMatrixResult result = constraints.toCMatrix();
    _specificationMatrix = result.C.to( _device );
    _specificationThresholds = result.thresholds.to( _device );
    _specificationBranchMapping = result.branchMapping;
    _specificationBranchSizes = result.branchSizes;
    _hasORBranches = result.hasORBranches;
    _hasSpecificationMatrix = true;

    log( Stringf( "[TorchModel] Specification matrix set: shape [%ld, %ld, %ld], %u constraints%s",
                  result.C.size( 0 ), result.C.size( 1 ), result.C.size( 2 ),
                  (unsigned)result.thresholds.size( 0 ),
                  result.hasORBranches
                      ? Stringf( ", %u OR branches", result.branchSizes.size() ).ascii()
                      : "" ) );
}

torch::Tensor TorchModel::getSpecificationMatrix() const
{
    if ( !_hasSpecificationMatrix )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No specification matrix set" );
    }
    return _specificationMatrix;
}

torch::Tensor TorchModel::getSpecificationThresholds() const
{
    if ( !_hasSpecificationMatrix )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No specification matrix set" );
    }
    return _specificationThresholds;
}

CMatrixResult TorchModel::getSpecificationMatrixResult() const
{
    if ( !_hasSpecificationMatrix )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No specification matrix set" );
    }
    CMatrixResult result;
    result.C = _specificationMatrix;
    result.thresholds = _specificationThresholds;
    result.branchMapping = _specificationBranchMapping;
    result.branchSizes = _specificationBranchSizes;
    result.hasORBranches = _hasORBranches;
    return result;
}

bool TorchModel::hasSpecificationMatrix() const
{
    return _hasSpecificationMatrix;
}

bool TorchModel::isSpecVerified( const torch::Tensor &lb, const torch::Tensor &ub ) const
{
    if ( !_hasSpecificationMatrix || !_specificationThresholds.defined() ||
         _specificationThresholds.numel() == 0 )
    {
        return false;
    }
    if ( !lb.defined() || lb.numel() == 0 || !ub.defined() || ub.numel() == 0 )
    {
        return false;
    }

    torch::Tensor lb_flat = lb.flatten();
    torch::Tensor ub_flat = ub.flatten();
    torch::Tensor thresholds = _specificationThresholds.to( lb_flat.device() ).flatten();

    if ( lb_flat.numel() != thresholds.numel() )
    {
        return false;
    }

    if ( _hasORBranches )
    {
        Vector<BranchResult> branchResults = OutputConstraintSet::evaluateORBranches(
            lb_flat, ub_flat, thresholds, _specificationBranchMapping, _specificationBranchSizes );
        for ( const auto &branch : branchResults )
        {
            if ( !branch.verified )
                return false;
        }
        return true;
    }

    // lb > threshold breaks the counterexample AND-conjunction
    return ( lb_flat > thresholds ).any().item<bool>();
}

BoundedTensor<torch::Tensor> TorchModel::compute_bounds(
    const BoundedTensor<torch::Tensor> &input_bounds, const torch::Tensor *specification_matrix,
    LunaConfiguration::AnalysisMethod method, bool bound_lower, bool bound_upper )
{

    log( "[TorchModel] compute_bounds() called - unified analysis entry point" );

    setInputBounds( input_bounds );

    if ( specification_matrix != nullptr )
    {
        setSpecificationMatrix( *specification_matrix );
        log( "[TorchModel] Specification matrix set from compute_bounds parameter" );
    }

    LunaConfiguration::ANALYSIS_METHOD = method;
    LunaConfiguration::COMPUTE_LOWER = bound_lower;
    LunaConfiguration::COMPUTE_UPPER = bound_upper;

    BoundedTensor<torch::Tensor> result;

    if ( method == LunaConfiguration::AnalysisMethod::CROWN )
    {
        log( "[TorchModel] Running CROWN analysis via compute_bounds" );
        result = runCROWN( input_bounds );
    }
    else if ( method == LunaConfiguration::AnalysisMethod::AlphaCROWN )
    {
        log( "[TorchModel] Running AlphaCROWN analysis via compute_bounds" );

        bool optimizeLower = bound_lower && LunaConfiguration::OPTIMIZE_LOWER;
        bool optimizeUpper = bound_upper && LunaConfiguration::OPTIMIZE_UPPER;

        log( Stringf( "[TorchModel] AlphaCROWN config: optimize_lower=%s, optimize_upper=%s",
                      optimizeLower ? "true" : "false", optimizeUpper ? "true" : "false" ) );

        result = runAlphaCROWN( input_bounds, optimizeLower, optimizeUpper );
    }
    else
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "Unknown analysis method" );
    }

    log( "[TorchModel] compute_bounds() completed successfully" );
    return result;
}

BoundedTensor<torch::Tensor> TorchModel::runCROWN()
{
    if ( !hasInputBounds() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                         "Input bounds must be set before running CROWN analysis" );
    }
    return runCROWN( _inputBounds );
}

BoundedTensor<torch::Tensor> TorchModel::runCROWN( const BoundedTensor<torch::Tensor> &inputBounds )
{
    log( "[TorchModel] Running CROWN analysis" );

    // populate cached shapes in Conv/BN nodes before backward propagation
    setInputBounds( inputBounds );
    cacheForwardShapesFromCenter();

    std::unique_ptr<CROWNAnalysis> crownAnalysis = std::make_unique<CROWNAnalysis>( this );

    crownAnalysis->setInputBounds( inputBounds );
    crownAnalysis->run();

    BoundedTensor<torch::Tensor> outputBounds = crownAnalysis->getOutputBounds();

    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> allBounds;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( crownAnalysis->hasConcreteBounds( i ) )
        {
            allBounds[i] = { crownAnalysis->getConcreteLowerBound( i ).detach().clone(),
                             crownAnalysis->getConcreteUpperBound( i ).detach().clone() };
        }
    }
    persistIntermediateBounds( allBounds );

    setFinalAnalysisBounds( outputBounds );

    log( "[TorchModel] CROWN analysis completed" );
    return outputBounds;
}

void TorchModel::cacheForwardShapesFromCenter()
{
    if ( !hasInputBounds() )
        return;

    torch::NoGradGuard no_grad;
    torch::Tensor lb = _inputBounds.lower().to( torch::kFloat32 );
    torch::Tensor ub = _inputBounds.upper().to( torch::kFloat32 );
    if ( !lb.defined() || !ub.defined() )
        return;

    torch::Tensor center = ( lb + ub ) / 2.0;

    // heuristic reshape to common image formats
    if ( center.numel() == 9408 )
    {
        center = center.view( { 1, 3, 56, 56 } );
    }
    else if ( center.numel() == 12288 )
    {
        center = center.view( { 1, 3, 64, 64 } );
    }
    else if ( center.numel() == 3072 )
    {
        center = center.view( { 1, 3, 32, 32 } );
    }
    else if ( center.numel() == 784 )
    {
        center = center.view( { 1, 1, 28, 28 } );
    }
    else
    {
        center = center.view( { 1, (long)center.numel() } );
    }

    try
    {
        (void)forwardAndStoreActivations( center );
    }
    catch ( ... )
    {
        // best-effort optimization; don't fail analysis
        return;
    }
}

BoundedTensor<torch::Tensor> TorchModel::runAlphaCROWN( bool optimizeLower, bool optimizeUpper )
{
    if ( !hasInputBounds() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                         "Input bounds must be set before running AlphaCROWN analysis" );
    }
    return runAlphaCROWN( _inputBounds, optimizeLower, optimizeUpper );
}

BoundedTensor<torch::Tensor>
TorchModel::runAlphaCROWN( const BoundedTensor<torch::Tensor> &inputBounds, bool optimizeLower,
                           bool optimizeUpper )
{
    log( "[TorchModel] Running AlphaCROWN analysis - Delegating to AlphaCROWNAnalysis" );

    // populate cached shapes in Conv/BN nodes before backward propagation
    setInputBounds( inputBounds );
    cacheForwardShapesFromCenter();

    std::unique_ptr<AlphaCROWNAnalysis> alphaCrownAnalysis =
        std::make_unique<AlphaCROWNAnalysis>( this );

    alphaCrownAnalysis->getCROWNAnalysis()->setInputBounds( inputBounds );

    if ( _hasPersistedAlphas )
        alphaCrownAnalysis->seedAlphas( _persistedAlphas );

    // early stop if initial CROWN pass already verifies the property
    if ( LunaConfiguration::STOP_CROWN_ON_VERIFIED )
    {
        alphaCrownAnalysis->initializeAlphaParameters();

        auto *crown = alphaCrownAnalysis->getCROWNAnalysis();
        unsigned outIdx = crown->getOutputIndex();
        if ( crown->hasConcreteBounds( outIdx ) )
        {
            torch::Tensor lb = crown->getConcreteLowerBound( outIdx );
            torch::Tensor ub = crown->getConcreteUpperBound( outIdx );
            if ( isSpecVerified( lb, ub ) )
            {
                std::cout << "Verified with initial CROWN!" << std::endl;
                BoundedTensor<torch::Tensor> result( lb, ub );
                setFinalAnalysisBounds( result );
                return result;
            }
        }
    }

    torch::Tensor finalLower, finalUpper;

    if ( optimizeUpper )
    {
        log( "[TorchModel] Requesting optimized upper bounds from AlphaCROWNAnalysis" );
        finalUpper =
            alphaCrownAnalysis->computeOptimizedBounds( LunaConfiguration::BoundSide::Upper );
        log( "[TorchModel] Received optimized upper bounds" );
    }

    if ( optimizeLower )
    {
        log( "[TorchModel] Requesting optimized lower bounds from AlphaCROWNAnalysis" );
        finalLower =
            alphaCrownAnalysis->computeOptimizedBounds( LunaConfiguration::BoundSide::Lower );
        log( "[TorchModel] Received optimized lower bounds" );
    }

    if ( !optimizeLower && !optimizeUpper )
    {
        log( "[TorchModel] No optimization requested, delegating to CROWN analysis" );
        return runCROWN( inputBounds );
    }

    if ( !optimizeLower )
    {
        log( "[TorchModel] Computing lower bounds with CROWN (not optimized)" );
        auto crownBounds = runCROWN( inputBounds );
        finalLower = crownBounds.lower();
    }
    if ( !optimizeUpper )
    {
        log( "[TorchModel] Computing upper bounds with CROWN (not optimized)" );
        auto crownBounds = runCROWN( inputBounds );
        finalUpper = crownBounds.upper();
    }

    persistAlphas( alphaCrownAnalysis->extractAlphas() );
    auto *crown = alphaCrownAnalysis->getCROWNAnalysis();
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> allBounds;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        if ( crown->hasConcreteBounds( i ) )
        {
            allBounds[i] = { crown->getConcreteLowerBound( i ).detach().clone(),
                             crown->getConcreteUpperBound( i ).detach().clone() };
        }
    }
    persistIntermediateBounds( allBounds );

    BoundedTensor<torch::Tensor> outputBounds( finalLower, finalUpper );

    setFinalAnalysisBounds( outputBounds );

    log( "[TorchModel] AlphaCROWN analysis completed" );
    return outputBounds;
}

void TorchModel::setCROWNBounds( unsigned nodeIndex, const BoundedTensor<torch::Tensor> &bounds )
{
    validateNodeIndex( nodeIndex );
    _crownBounds[nodeIndex] = bounds;
    log( Stringf( "[TorchModel] Set CROWN bounds for node %u", nodeIndex ) );
}

void TorchModel::setAlphaCROWNBounds( unsigned nodeIndex,
                                      const BoundedTensor<torch::Tensor> &bounds )
{
    validateNodeIndex( nodeIndex );
    _alphaCrownBounds[nodeIndex] = bounds;
    log( Stringf( "[TorchModel] Set AlphaCROWN bounds for node %u", nodeIndex ) );
}

BoundedTensor<torch::Tensor> TorchModel::getCROWNBounds( unsigned nodeIndex ) const
{
    validateNodeIndex( nodeIndex );
    if ( !_crownBounds.exists( nodeIndex ) )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                         Stringf( "No CROWN bounds available for node %u", nodeIndex ).ascii() );
    }
    return _crownBounds.at( nodeIndex );
}

BoundedTensor<torch::Tensor> TorchModel::getAlphaCROWNBounds( unsigned nodeIndex ) const
{
    validateNodeIndex( nodeIndex );
    if ( !_alphaCrownBounds.exists( nodeIndex ) )
    {
        throw LunaError(
            LunaError::INVALID_MODEL_STRUCTURE,
            Stringf( "No AlphaCROWN bounds available for node %u", nodeIndex ).ascii() );
    }
    return _alphaCrownBounds.at( nodeIndex );
}

bool TorchModel::hasCROWNBounds( unsigned nodeIndex ) const
{
    return nodeIndex < _nodes.size() && _crownBounds.exists( nodeIndex );
}

bool TorchModel::hasAlphaCROWNBounds( unsigned nodeIndex ) const
{
    return nodeIndex < _nodes.size() && _alphaCrownBounds.exists( nodeIndex );
}

void TorchModel::setFinalAnalysisBounds( const BoundedTensor<torch::Tensor> &bounds )
{
    _finalAnalysisBounds = bounds;
    _hasFinalAnalysisBounds = true;
    log( "[TorchModel] Set final analysis bounds" );
}

BoundedTensor<torch::Tensor> TorchModel::getFinalAnalysisBounds() const
{
    if ( !_hasFinalAnalysisBounds )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE, "No final analysis bounds available" );
    }
    return _finalAnalysisBounds;
}

bool TorchModel::hasFinalAnalysisBounds() const
{
    return _hasFinalAnalysisBounds;
}

void TorchModel::validateNodeIndex( unsigned nodeIndex ) const
{
    if ( nodeIndex >= _nodes.size() )
    {
        throw LunaError( LunaError::INVALID_MODEL_STRUCTURE,
                         Stringf( "Node index %u out of bounds for model with %u nodes", nodeIndex,
                                  _nodes.size() )
                             .ascii() );
    }
}

void TorchModel::loadState(
    const BoundedTensor<torch::Tensor> &inputBounds,
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &intermediateBounds,
    const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> &alphas )
{
    setInputBounds( inputBounds );
    persistIntermediateBounds( intermediateBounds );
    persistAlphas( alphas );
    log( "[TorchModel] loadState() completed" );
}

void TorchModel::loadProp( const torch::Tensor &C )
{
    torch::Tensor spec = C;
    if ( spec.dim() == 2 )
        spec = spec.unsqueeze( 1 );
    setSpecificationMatrix( spec );
    log( "[TorchModel] loadProp() completed" );
}

TorchModel::BaBState TorchModel::getState() const
{
    BaBState state;
    for ( unsigned i = 0; i < _nodes.size(); ++i )
    {
        auto &node = _nodes[i];
        if ( node && node->hasBounds() )
        {
            state.allBounds[i] = { node->getLower().detach().clone(),
                                   node->getUpper().detach().clone() };
        }
        else if ( _concreteBounds.exists( i ) )
        {
            state.allBounds[i] = { _concreteBounds[i].lower().detach().clone(),
                                   _concreteBounds[i].upper().detach().clone() };
        }
    }
    state.alphas = _persistedAlphas;
    return state;
}

void TorchModel::persistAlphas(
    const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> &alphas )
{
    _persistedAlphas.clear();
    for ( auto &[nodeIdx, perStart] : alphas )
    {
        for ( auto &[startKey, ap] : perStart )
        {
            AlphaParameters copy;
            copy.alpha = ap.alpha.defined() ? ap.alpha.detach().clone() : torch::Tensor();
            copy.unstableMask =
                ap.unstableMask.defined() ? ap.unstableMask.detach().clone() : torch::Tensor();
            copy.unstableIndices = ap.unstableIndices.defined()
                                       ? ap.unstableIndices.detach().clone()
                                       : torch::Tensor();
            copy.specDim = ap.specDim;
            copy.batchDim = ap.batchDim;
            copy.outDim = ap.outDim;
            copy.numUnstable = ap.numUnstable;
            copy.requiresGrad = ap.requiresGrad;
            copy.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;
            _persistedAlphas[nodeIdx][startKey] = std::move( copy );
        }
    }
    _hasPersistedAlphas = !_persistedAlphas.empty();
}

void TorchModel::persistIntermediateBounds(
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &bounds )
{
    _persistedIntermediateBounds.clear();
    for ( auto &[nodeIdx, pair] : bounds )
    {
        _persistedIntermediateBounds[nodeIdx] = { pair.first.detach().clone(),
                                                  pair.second.detach().clone() };
    }
    _hasPersistedIntermediateBounds = !_persistedIntermediateBounds.empty();
}

bool TorchModel::hasPersistedAlphas() const
{
    return _hasPersistedAlphas;
}

bool TorchModel::hasPersistedIntermediateBounds() const
{
    return _hasPersistedIntermediateBounds;
}

const std::unordered_map<unsigned, std::unordered_map<std::string, AlphaParameters>> &
TorchModel::getPersistedAlphas() const
{
    return _persistedAlphas;
}

const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> &
TorchModel::getPersistedIntermediateBounds() const
{
    return _persistedIntermediateBounds;
}

void TorchModel::clearPersistedState()
{
    _persistedAlphas.clear();
    _hasPersistedAlphas = false;
    _persistedIntermediateBounds.clear();
    _hasPersistedIntermediateBounds = false;
    log( "[TorchModel] clearPersistedState() completed" );
}

} // namespace NLR
