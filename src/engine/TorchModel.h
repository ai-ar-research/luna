/*********************                                                        */
/*! \file TorchModel.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __TorchModel_h__
#define __TorchModel_h__

#include "Map.h"
#include "Set.h"
#include "MString.h"
#include "Vector.h"
#include "nodes/BoundedTorchNode.h"
#include "nodes/BoundedInputNode.h"
#include "nodes/BoundedLinearNode.h"
#include "nodes/BoundedReLUNode.h"
#include "nodes/BoundedIdentityNode.h"
#include "nodes/BoundedConstantNode.h"
#include "nodes/BoundedReshapeNode.h"
#include "input_parsers/OutputConstraint.h"
#include "configuration/LunaConfiguration.h"

class CROWNAnalysis;
class AlphaCROWNAnalysis;

// avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include "AlphaParameters.h"

#include <torch/torch.h>
#include <memory>
#include <unordered_map>

namespace NLR {

class TorchModel {
public:
    TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes,
               const Vector<unsigned>& inputIndices,
               unsigned outputIndex,
               const Map<unsigned, Vector<unsigned>>& dependencies);

    TorchModel(const String& onnxPath);

    TorchModel(const String& onnxPath,
               const String& vnnlibPath);

    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations,
                         const Map<unsigned, torch::Tensor>& inputs);

    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const torch::Tensor& input);
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs);

    unsigned getInputSize() const { return _input_size; }
    unsigned getOutputSize() const { return _output_size; }
    unsigned getNumNodes() const { return _nodes.size(); }
    torch::Device getDevice() const { return _device; }
    void moveToDevice(const torch::Device& device);

    const Vector<std::shared_ptr<BoundedTorchNode>>& getNodes() const { return _nodes; }
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    Vector<unsigned> getAllNodeIndices() const;
    Vector<unsigned> getNodesByType(NodeType type) const;
    const Vector<unsigned>& getInputIndices() const { return _inputIndices; }
    unsigned getOutputIndex() const { return _outputIndex; }

    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);

    void setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds);
    void clearConcreteBounds();
    BoundedTensor<torch::Tensor> getConcreteBounds(unsigned nodeIndex) const;
    bool hasConcreteBounds(unsigned nodeIndex) const;

    BoundedTensor<torch::Tensor> getInputBounds() const;
    bool hasInputBounds() const;
    torch::Tensor getInputLowerBounds() const;
    torch::Tensor getInputUpperBounds() const;

    void setSpecificationMatrix(const torch::Tensor& specMatrix);
    void setSpecificationFromConstraints(const OutputConstraintSet& constraints);
    torch::Tensor getSpecificationMatrix() const;
    torch::Tensor getSpecificationThresholds() const;
    CMatrixResult getSpecificationMatrixResult() const;
    bool hasSpecificationMatrix() const;
    bool isSpecVerified(const torch::Tensor& lb, const torch::Tensor& ub) const;

    BoundedTensor<torch::Tensor> compute_bounds(
        const BoundedTensor<torch::Tensor>& input_bounds,
        const torch::Tensor* specification_matrix = nullptr,
        LunaConfiguration::AnalysisMethod method = LunaConfiguration::AnalysisMethod::CROWN,
        bool bound_lower = true,
        bool bound_upper = true
    );

    BoundedTensor<torch::Tensor> runCROWN();
    BoundedTensor<torch::Tensor> runCROWN(const BoundedTensor<torch::Tensor>& inputBounds);

    BoundedTensor<torch::Tensor> runAlphaCROWN(bool optimizeLower = true, bool optimizeUpper = false);
    BoundedTensor<torch::Tensor> runAlphaCROWN(const BoundedTensor<torch::Tensor>& inputBounds,
                                                bool optimizeLower = true, bool optimizeUpper = false);

    void setCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    void setAlphaCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getCROWNBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getAlphaCROWNBounds(unsigned nodeIndex) const;
    bool hasCROWNBounds(unsigned nodeIndex) const;
    bool hasAlphaCROWNBounds(unsigned nodeIndex) const;

    void setFinalAnalysisBounds(const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getFinalAnalysisBounds() const;
    bool hasFinalAnalysisBounds() const;

    struct BaBState {
        std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>> allBounds;
        std::unordered_map<unsigned,
            std::unordered_map<std::string, AlphaParameters>> alphas;
    };

    void loadState(
        const BoundedTensor<torch::Tensor>& inputBounds,
        const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& intermediateBounds,
        const std::unordered_map<unsigned,
            std::unordered_map<std::string, AlphaParameters>>& alphas);
    void loadProp(const torch::Tensor& C);
    BaBState getState() const;

    void persistAlphas(
        const std::unordered_map<unsigned,
            std::unordered_map<std::string, AlphaParameters>>& alphas);
    void persistIntermediateBounds(
        const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& bounds);
    bool hasPersistedAlphas() const;
    bool hasPersistedIntermediateBounds() const;
    const std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>>& getPersistedAlphas() const;
    const std::unordered_map<unsigned,
        std::pair<torch::Tensor, torch::Tensor>>& getPersistedIntermediateBounds() const;
    void clearPersistedState();

    const Map<unsigned, Vector<unsigned>>& getDependenciesMap() const { return _dependencies; }

    void buildDependencyGraph();
    void buildDependents();
    void computeDegrees();

    Vector<unsigned> topologicalSort() const;
    Vector<unsigned> getRoots() const;
    Vector<unsigned> getLeaves() const;
    Vector<unsigned> getDependents(unsigned nodeIndex) const;
    Vector<unsigned> getDependencies(unsigned nodeIndex) const;

    unsigned getDegreeOut(unsigned nodeIndex) const;
    unsigned getDegreeIn(unsigned nodeIndex) const;
    void resetProcessingState();
    bool isProcessed(unsigned nodeIndex) const;
    void markProcessed(unsigned nodeIndex);

    void log(const String& message) const;

private:
    Vector<std::shared_ptr<BoundedTorchNode>> _nodes;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<unsigned>> _dependencies;

    Map<unsigned, Vector<unsigned>> _dependents;
    Map<unsigned, unsigned> _degreeOut;
    Map<unsigned, unsigned> _degreeIn;
    Map<unsigned, bool> _processed;

    unsigned _input_size;
    unsigned _output_size;

    BoundedTensor<torch::Tensor> _inputBounds;
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;

    torch::Tensor _specificationMatrix;
    torch::Tensor _specificationThresholds;
    Vector<unsigned> _specificationBranchMapping;
    Vector<unsigned> _specificationBranchSizes;
    bool _hasSpecificationMatrix;
    bool _hasORBranches;
    Map<unsigned, BoundedTensor<torch::Tensor>> _crownBounds;
    Map<unsigned, BoundedTensor<torch::Tensor>> _alphaCrownBounds;
    BoundedTensor<torch::Tensor> _finalAnalysisBounds;
    bool _hasFinalAnalysisBounds;

    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> _persistedAlphas;
    bool _hasPersistedAlphas{false};
    std::unordered_map<unsigned,
        std::pair<torch::Tensor, torch::Tensor>> _persistedIntermediateBounds;
    bool _hasPersistedIntermediateBounds{false};

    void validateNodeIndex(unsigned nodeIndex) const;
    void cacheForwardShapesFromCenter();

    torch::Device _device;
};

} // namespace NLR

#endif // __TorchModel_h__
