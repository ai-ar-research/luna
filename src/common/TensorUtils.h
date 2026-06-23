/*********************                                                        */
/*! \file TensorUtils.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __TensorUtils_h__
#define __TensorUtils_h__

#include "configuration/LunaConfiguration.h"
#include <torch/torch.h>

namespace LirpaTensorUtils {

inline torch::TensorOptions defaultOptions()
{
    return torch::TensorOptions().dtype(torch::kFloat32).device(LunaConfiguration::getDevice());
}

inline torch::Tensor toDevice(const torch::Tensor &tensor)
{
    return tensor.to(LunaConfiguration::getDevice());
}

inline torch::Tensor zerosLikeOnDevice(const torch::Tensor &reference)
{
    return torch::zeros_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

inline torch::Tensor onesLikeOnDevice(const torch::Tensor &reference)
{
    return torch::ones_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

inline torch::Tensor emptyLikeOnDevice(const torch::Tensor &reference)
{
    return torch::empty_like(reference, reference.options().device(LunaConfiguration::getDevice()));
}

} // namespace LirpaTensorUtils

#endif // __TensorUtils_h__
