#include "BoundedAlphaOptimizedNode.h"
#include "AlphaCROWNAnalysis.h"

namespace NLR {

// Optimization side queries (delegated to AlphaCROWNAnalysis)
bool BoundedAlphaOptimizeNode::isOptimizingLower() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingLower() : true;  // Default to lower
}

bool BoundedAlphaOptimizeNode::isOptimizingUpper() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingUpper() : false; // Default to not upper
}

bool BoundedAlphaOptimizeNode::isOptimizingBoth() const {
    return false;  // Always false - removed "both" mode in favor of binary selection
}

} // namespace NLR