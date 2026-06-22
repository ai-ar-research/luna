/*********************                                                        */
/*! \file BoundedAlphaOptimizedNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedAlphaOptimizedNode.h"
#include "AlphaCROWNAnalysis.h"

namespace NLR {

bool BoundedAlphaOptimizeNode::isOptimizingLower() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingLower() : true;
}

bool BoundedAlphaOptimizeNode::isOptimizingUpper() const {
    return _alphaCrownAnalysis ? _alphaCrownAnalysis->isOptimizingUpper() : false;
}

bool BoundedAlphaOptimizeNode::isOptimizingBoth() const {
    return false;
}

} // namespace NLR
