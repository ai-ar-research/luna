/*********************                                                        */
/*! \file BoundedFlattenNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedFlattenNode.h"

namespace NLR {

BoundedFlattenNode::BoundedFlattenNode(const Operations::FlattenWrapper& flatten_module)
    : _flatten_module(flatten_module) {
    _nodeName = "flatten";
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
}

torch::Tensor BoundedFlattenNode::forward(const torch::Tensor& input) {
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    torch::Tensor output = _flatten_module.forward(input);

    _input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }

    return output;
}

// No-op for bound propagation: downstream nodes expect flattened features axis
void BoundedFlattenNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedFlattenNode expects at least one input");
    }

    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedFlattenNode: Patches propagation not implemented (convert to matrix)");
    }

    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();

    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(lA), BoundA(uA)));

    torch::Device device = _device;
    auto zero_bias_like_A = [device](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
            return torch::zeros({1, 1}, options);
        }
        if (A.dim() == 3) {
            // [spec, batch, features] -> bias [spec, batch]
            return torch::zeros({A.size(0), A.size(1)}, A.options());
        }
        if (A.dim() == 2) {
            // [spec, features] -> bias [spec, 1]
            return torch::zeros({A.size(0), 1}, A.options());
        }
        return torch::zeros({1, 1}, A.options());
    };

    lbias = zero_bias_like_A(lA);
    ubias = zero_bias_like_A(uA);
}

BoundedTensor<torch::Tensor> BoundedFlattenNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Flatten module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    torch::Tensor flattenedLower = _flatten_module.forward(inputLowerBound);
    torch::Tensor flattenedUpper = _flatten_module.forward(inputUpperBound);

    return BoundedTensor<torch::Tensor>(flattenedLower, flattenedUpper);
}

void NLR::BoundedFlattenNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedFlattenNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
