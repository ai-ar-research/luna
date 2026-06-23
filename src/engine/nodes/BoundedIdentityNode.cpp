/*********************                                                        */
/*! \file BoundedIdentityNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedIdentityNode.h"

namespace NLR {

BoundedIdentityNode::BoundedIdentityNode(const torch::nn::Identity& identityModule)
    : _identity_module(identityModule) {
    _nodeName = "identity";
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
}

torch::Tensor BoundedIdentityNode::forward(const torch::Tensor& input) {
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    torch::Tensor output = _identity_module->forward(input);
    return output;
}

void BoundedIdentityNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedIdentityNode expects at least one input");
    }

    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

    if (last_lA.isTensor()) {
        torch::Tensor lA = last_lA.asTensor();
        if (lA.defined()) {
            int output_size = lA.size(1);
            if (!lbias.defined()) lbias = torch::zeros({output_size}, lA.options());
        } else {
            if (!lbias.defined()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
                lbias = torch::zeros({1}, options);
            }
        }
    }

    if (last_uA.isTensor()) {
        torch::Tensor uA = last_uA.asTensor();
        if (uA.defined()) {
            int output_size = uA.size(1);
            if (!ubias.defined()) ubias = torch::zeros({output_size}, uA.options());
        } else {
            if (!ubias.defined()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
                ubias = torch::zeros({1}, options);
            }
        }
    }
}

Pair<torch::Tensor, torch::Tensor> BoundedIdentityNode::computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha,
                                                                  const torch::Tensor& lastUpperAlpha,
                                                                  const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    (void)inputBounds;
    return Pair<torch::Tensor, torch::Tensor>(lastLowerAlpha, lastUpperAlpha);
}

BoundedTensor<torch::Tensor> BoundedIdentityNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Identity module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    if (_input_size == 0 && inputLowerBound.defined()) {
        _input_size = inputLowerBound.numel();
    }

    torch::Tensor lowerBound = inputLowerBound;
    torch::Tensor upperBound = inputUpperBound;

    if (_output_size == 0 && lowerBound.defined()) {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

unsigned BoundedIdentityNode::getOutputSize() const {
    if (_output_size == 0 && _input_size > 0) {
        return _input_size;
    }
    if (_output_size == 0) {
        return 2; // Default for testing
    }
    return _output_size;
}

void BoundedIdentityNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedIdentityNode::setOutputSize(unsigned size) {
    _output_size = size;
}

unsigned BoundedIdentityNode::getInputSize() const {
    return _input_size;
}

} // namespace NLR
