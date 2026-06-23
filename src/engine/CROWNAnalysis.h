/*********************                                                        */
/*! \file CROWNAnalysis.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __CROWNAnalysis_h__
#define __CROWNAnalysis_h__

#include "TorchModel.h"
#include "nodes/BoundedTorchNode.h"
#include "BoundedTensor.h"
#include "BoundResult.h"
#include "Map.h"
#include "Vector.h"
#include "Set.h"
#include "Queue.h"
#include "input_parsers/OutputConstraint.h"
#include "configuration/LunaConfiguration.h"

#include <torch/torch.h>
#include <memory>
#include <unordered_map>

namespace NLR {

struct CrownStartContext {
    std::string start_key;
    int spec_dim{1};
};

struct AlphaStartCache {
    Vector<unsigned> unstableIndices;
    bool sparseMode{false};
    unsigned nodeSize{0};
    bool initialized{false};
};

class CROWNAnalysis
{
public:
    CROWNAnalysis( TorchModel *torchModel );
    ~CROWNAnalysis();

    // enableGradients: needed for alpha-CROWN optimization
    void run(bool enableGradients = false);

    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    unsigned getInputSize() const;
    unsigned getOutputSize() const;
    unsigned getOutputIndex() const;

    torch::Tensor getIBPLowerBound(unsigned nodeIndex);
    torch::Tensor getIBPUpperBound(unsigned nodeIndex);
    torch::Tensor getCrownLowerBound(unsigned nodeIndex) const;
    torch::Tensor getCrownUpperBound(unsigned nodeIndex) const;
    bool hasIBPBounds(unsigned nodeIndex);
    bool hasCrownBounds(unsigned nodeIndex);
    unsigned getNumNodes() const;

    torch::Tensor getConcreteLowerBound(unsigned nodeIndex);
    torch::Tensor getConcreteUpperBound(unsigned nodeIndex);
    bool hasConcreteBounds(unsigned nodeIndex);

    BoundedTensor<torch::Tensor> getOutputBounds() const;
    BoundedTensor<torch::Tensor> getOutputIBPBounds() const;

    TorchModel* getModel() const { return _torchModel; }

    Vector<BoundedTensor<torch::Tensor>> getInputBoundsForNode(unsigned nodeIndex);

    void resetProcessingState();
    void clearConcreteBounds();
    void clearAllNodeBounds();
    void markProcessed(unsigned nodeIndex);
    bool isProcessed(unsigned nodeIndex) const;

    bool checkIBPFirstLinear(unsigned nodeIndex);
    bool isFirstLinearLayer(unsigned nodeIndex);

    bool checkIBPIntermediate(unsigned nodeIndex);

    void computeIntermediateBoundsLazy(unsigned nodeIndex);

    void checkPriorBounds(unsigned nodeIndex);

    void computeIBPForNode(unsigned nodeIndex);


    void computeIBPBounds();
    void computeCrownBackwardPropagation();
    void concretizeBounds();

    void computeForwardPassValues();

    torch::Tensor computeConcreteLowerBound(const torch::Tensor& lA, const torch::Tensor& lBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);
    torch::Tensor computeConcreteUpperBound(const torch::Tensor& uA, const torch::Tensor& uBias,
                                           const torch::Tensor& xLower, const torch::Tensor& xUpper);


    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    BoundedTensor<torch::Tensor> getNodeIBPBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeCrownBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getNodeConcreteBounds(unsigned nodeIndex) const;

    BoundA addA(const BoundA& A1, const BoundA& A2);
    void addBound(unsigned nodeIndex, const BoundA& lA, const BoundA& uA);
    void addBias(unsigned nodeIndex, const torch::Tensor& lBias, const torch::Tensor& uBias);

    // C=nullptr queries TorchModel for spec matrix; otherwise uses identity
    void backwardFrom(unsigned startIndex, const Vector<unsigned>& unstableIndices = {}, const torch::Tensor* C = nullptr);

    void clearCrownState();

    const CrownStartContext& currentStart() const { return _cur; }
    const std::string& currentStartKey() const { return _currentStartKey; }
    int currentStartSpecDim() const { return _currentStartSpecDim; }
    const Vector<unsigned>& currentStartSpecIndices() const { return _currentStartSpecIndices; }

    void _setCurrentStart(const std::string& key, int specDim) {
        _cur.start_key = key;
        _cur.spec_dim = specDim;
        _currentStartKey = key;
        _currentStartSpecDim = specDim;
    }

    void setAlphaStartCacheEnabled(bool enabled) { _alphaStartCacheEnabled = enabled; }
    void clearAlphaStartCache() { _alphaStartCache.clear(); }
    bool getAlphaStartCacheInfo(const std::string& key,
                                Vector<unsigned>& unstableIndices,
                                bool& sparseMode,
                                unsigned& nodeSize) const;

    void concretizeNode(unsigned startIndex, const Vector<unsigned>& unstableIndices = {});

    // STE stabilization: clamps intermediate bounds to reference while preserving gradient flow
    void setReferenceBounds(const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& ref);
    void clearReferenceBounds();

    bool needsCROWNBounds(unsigned nodeIndex);

private:
    TorchModel *_torchModel;

    Map<unsigned, std::shared_ptr<BoundedTorchNode>> _nodes;

    Map<unsigned, BoundA> _lA;
    Map<unsigned, BoundA> _uA;

    Map<unsigned, torch::Tensor> _lowerBias;
    Map<unsigned, torch::Tensor> _upperBias;

    Map<unsigned, BoundedTensor<torch::Tensor>> _ibpBounds;

    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    Map<unsigned, torch::Tensor> _forwardPassValues;

    bool _hasCenterActivations{false};
    Map<unsigned, torch::Tensor> _centerActivations;

    bool _foundFirstUnsound{false};
    unsigned _firstUnsoundNode{0};

    std::unordered_map<std::string, AlphaStartCache> _alphaStartCache;
    bool _alphaStartCacheEnabled{false};
    Vector<unsigned> _currentStartSpecIndices;

    void computeConcreteBounds(const torch::Tensor& lA, const torch::Tensor& uA,
                              const torch::Tensor& lBias, const torch::Tensor& uBias,
                              const torch::Tensor& nodeLower, const torch::Tensor& nodeUpper,
                              torch::Tensor& concreteLower, torch::Tensor& concreteUpper);

    void ensureCenterActivations();
    torch::Tensor buildCenterInputForForward() const;

    void log( const String &message );
    std::string nodeTypeToString(NodeType type) {
        switch (type) {
            case NodeType::INPUT: return "INPUT";
            case NodeType::CONSTANT: return "CONSTANT";
            case NodeType::LINEAR: return "LINEAR";
            case NodeType::RELU: return "RELU";
            case NodeType::RESHAPE: return "RESHAPE";
            case NodeType::FLATTEN: return "FLATTEN";
            case NodeType::IDENTITY: return "IDENTITY";
            case NodeType::ADD: return "ADD";
            case NodeType::SUB: return "SUB";
            case NodeType::CONV: return "CONV";
            case NodeType::BATCHNORM: return "BATCHNORM";
            case NodeType::SIGMOID: return "SIGMOID";
            default: return "UNKNOWN";
        }
    }

    // Transforms C from (batch, spec) to (spec, batch, *output_shape)
    torch::Tensor preprocessC(const torch::Tensor& C, unsigned startIndex);

    CrownStartContext _cur;
    std::string _currentStartKey;
    int _currentStartSpecDim{1};

    Set<unsigned> _nodesNeedingBounds;

    Map<unsigned, Pair<BoundA, torch::Tensor>> _intermediateA;
    Map<unsigned, Pair<BoundA, torch::Tensor>> _intermediateAUpper;

    // Maps nodeIndex -> (lower, upper) detached reference tensors
    std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> _referenceBounds;

};

} // namespace NLR

#endif // __CROWNAnalysis_h__
