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

// Forward declarations to avoid circular dependency
class CROWNAnalysis;
class AlphaCROWNAnalysis;

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <memory>

namespace NLR {

class TorchModel {
public:
    // Traditional constructor with pre-built nodes
    TorchModel(const Vector<std::shared_ptr<BoundedTorchNode>>& nodes,
               const Vector<unsigned>& inputIndices,
               unsigned outputIndex,
               const Map<unsigned, Vector<unsigned>>& dependencies);

    // Constructor that loads from ONNX file (mirrors auto_LiRPA BoundedModule)
    TorchModel(const String& onnxPath);

    // Constructor that loads from ONNX file and VNN-LIB file for input bounds
    TorchModel(const String& onnxPath,
               const String& vnnlibPath);

    // Forward pass through the entire model
    torch::Tensor forward(const torch::Tensor& input);
    torch::Tensor forward(unsigned nodeIndex, Map<unsigned, torch::Tensor>& activations, 
                         const Map<unsigned, torch::Tensor>& inputs);

    
    
    // forward pass that returns activations for all nodes
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const torch::Tensor& input);
    Map<unsigned, torch::Tensor> forwardAndStoreActivations(const Map<unsigned, torch::Tensor>& inputs);            

    // Get model information
    unsigned getInputSize() const { return _input_size; }
    unsigned getOutputSize() const { return _output_size; }
    unsigned getNumNodes() const { return _nodes.size(); }
    torch::Device getDevice() const { return _device; }
    void moveToDevice(const torch::Device& device);
    
    // Access to nodes
    const Vector<std::shared_ptr<BoundedTorchNode>>& getNodes() const { return _nodes; }
    std::shared_ptr<BoundedTorchNode> getNode(unsigned index) const;
    Vector<unsigned> getAllNodeIndices() const;
    Vector<unsigned> getNodesByType(NodeType type) const;
    const Vector<unsigned>& getInputIndices() const { return _inputIndices; }
    unsigned getOutputIndex() const { return _outputIndex; }
    
    // PRIMARY BOUND MANAGEMENT INTERFACE
    void setInputBounds(const BoundedTensor<torch::Tensor>& inputBounds);
    
    // CONCRETE BOUND STORAGE (for CROWN analysis to call)
    void setConcreteBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& concreteBounds);
    void clearConcreteBounds();  // Clear all cached concrete bounds
    // FOR TESTING AND OUTPUTTING
    BoundedTensor<torch::Tensor> getConcreteBounds(unsigned nodeIndex) const;
    bool hasConcreteBounds(unsigned nodeIndex) const;
    
    // Input bound access
    BoundedTensor<torch::Tensor> getInputBounds() const;
    bool hasInputBounds() const;
    torch::Tensor getInputLowerBounds() const;
    torch::Tensor getInputUpperBounds() const;

    // SPECIFICATION MATRIX MANAGEMENT
    void setSpecificationMatrix(const torch::Tensor& specMatrix);
    void setSpecificationFromConstraints(const OutputConstraintSet& constraints);
    torch::Tensor getSpecificationMatrix() const;
    torch::Tensor getSpecificationThresholds() const;
    CMatrixResult getSpecificationMatrixResult() const;
    bool hasSpecificationMatrix() const;
    bool isSpecVerified(const torch::Tensor& lb, const torch::Tensor& ub) const;

    // UNIFIED ANALYSIS ENTRY METHOD (mirrors auto_LiRPA's compute_bounds)
    // Configuration is read from LunaConfiguration static members
    BoundedTensor<torch::Tensor> compute_bounds(
        const BoundedTensor<torch::Tensor>& input_bounds,
        const torch::Tensor* specification_matrix = nullptr,
        LunaConfiguration::AnalysisMethod method = LunaConfiguration::AnalysisMethod::CROWN,
        bool bound_lower = true,
        bool bound_upper = true
    );

    // ANALYSIS ENTRY METHODS
    BoundedTensor<torch::Tensor> runCROWN();
    BoundedTensor<torch::Tensor> runCROWN(const BoundedTensor<torch::Tensor>& inputBounds);

    // NEW ALPHA-CROWN ENTRY METHODS (REFACTORED)
    // TorchModel controls optimization flow and calls AlphaCROWN for bound computation
    BoundedTensor<torch::Tensor> runAlphaCROWN(bool optimizeLower = true, bool optimizeUpper = false);
    BoundedTensor<torch::Tensor> runAlphaCROWN(const BoundedTensor<torch::Tensor>& inputBounds,
                                                bool optimizeLower = true, bool optimizeUpper = false);


    // ANALYSIS BOUNDS STORAGE
    void setCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    void setAlphaCROWNBounds(unsigned nodeIndex, const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getCROWNBounds(unsigned nodeIndex) const;
    BoundedTensor<torch::Tensor> getAlphaCROWNBounds(unsigned nodeIndex) const;
    bool hasCROWNBounds(unsigned nodeIndex) const;
    bool hasAlphaCROWNBounds(unsigned nodeIndex) const;

    // FINAL ANALYSIS BOUNDS (output bounds from most recent analysis)
    void setFinalAnalysisBounds(const BoundedTensor<torch::Tensor>& bounds);
    BoundedTensor<torch::Tensor> getFinalAnalysisBounds() const;
    bool hasFinalAnalysisBounds() const;
    
    // Dependency graph access
    const Map<unsigned, Vector<unsigned>>& getDependenciesMap() const { return _dependencies; }

    // Full graph 
    void buildDependencyGraph();
    void buildDependents();
    void computeDegrees();

    // Traversal 
    Vector<unsigned> topologicalSort() const;
    Vector<unsigned> getRoots() const;
    Vector<unsigned> getLeaves() const;
    Vector<unsigned> getDependents(unsigned nodeIndex) const;
    Vector<unsigned> getDependencies(unsigned nodeIndex) const;

    // Degree and Processing states
    unsigned getDegreeOut(unsigned nodeIndex) const;
    unsigned getDegreeIn(unsigned nodeIndex) const;
    void resetProcessingState();
    bool isProcessed(unsigned nodeIndex) const;
    void markProcessed(unsigned nodeIndex);

    // Logging
    void log(const String& message) const;

private:
    Vector<std::shared_ptr<BoundedTorchNode>> _nodes;
    Vector<unsigned> _inputIndices;
    unsigned _outputIndex;
    Map<unsigned, Vector<unsigned>> _dependencies;

    // Graph traversal state
    Map<unsigned, Vector<unsigned>> _dependents;
    Map<unsigned, unsigned> _degreeOut;
    Map<unsigned, unsigned> _degreeIn;
    Map<unsigned, bool> _processed;

    // Model dimensions
    unsigned _input_size;
    unsigned _output_size;

    BoundedTensor<torch::Tensor> _inputBounds;      // Input bounds for the model
    Map<unsigned, BoundedTensor<torch::Tensor>> _concreteBounds;  // CROWN concrete bounds

    // SPECIFICATION AND ANALYSIS STORAGE
    torch::Tensor _specificationMatrix;            // Specification matrix (C matrix)
    torch::Tensor _specificationThresholds;        // Thresholds for each constraint
    Vector<unsigned> _specificationBranchMapping;  // Branch mapping for OR disjunctions (row -> branch)
    Vector<unsigned> _specificationBranchSizes;    // Branch sizes for OR disjunctions (branch -> size)
    bool _hasSpecificationMatrix;                  // Whether specification matrix is set
    bool _hasORBranches;                           // Whether specification has OR branches
    Map<unsigned, BoundedTensor<torch::Tensor>> _crownBounds;      // CROWN analysis bounds
    Map<unsigned, BoundedTensor<torch::Tensor>> _alphaCrownBounds; // AlphaCROWN analysis bounds
    BoundedTensor<torch::Tensor> _finalAnalysisBounds;             // Final analysis output bounds
    bool _hasFinalAnalysisBounds;                  // Whether final bounds are set

    // Configuration is accessed via LunaConfiguration static members
    
    // error checking
    void validateNodeIndex(unsigned nodeIndex) const;

    // Optional: run a single forward pass at the input-box center to cache per-node shapes
    // inside nodes (e.g., Conv/BatchNorm), so backward/CROWN code doesn't rely on heuristics.
    void cacheForwardShapesFromCenter();

    torch::Device _device;
};

} // namespace NLR

#endif // __TorchModel_h__ 