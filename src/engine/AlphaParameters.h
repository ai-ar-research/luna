/*********************                                                        */
/*! \file AlphaParameters.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __AlphaParameters_h__
#define __AlphaParameters_h__

#include <torch/torch.h>

namespace NLR {

struct AlphaParameters {
    torch::Tensor alpha;
    torch::Tensor unstableMask;
    torch::Tensor unstableIndices;
    int specDim{0};
    int batchDim{1};
    int outDim{0};
    int numUnstable{0};
    bool requiresGrad{true};
    bool hasSpecDefaultSlot{false};
};

} // namespace NLR

#endif // __AlphaParameters_h__
