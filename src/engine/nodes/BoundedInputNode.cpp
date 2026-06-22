/*********************                                                        */
/*! \file BoundedInputNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedInputNode.h"

NLR::BoundedInputNode::BoundedInputNode(unsigned inputIndex, unsigned inputSize, const String& name)
    : _inputIndex(inputIndex) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = inputSize;
    _output_size = inputSize;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
    torch::Tensor lower = torch::zeros({inputSize}, options);
    torch::Tensor upper = torch::ones({inputSize}, options);
    _inputBounds = BoundedTensor<torch::Tensor>(lower, upper);
}

torch::Tensor NLR::BoundedInputNode::forward(const torch::Tensor& input) {
    if (input.defined() && input.numel() > 0) {
        return input.to(torch::kFloat32);
    }

    torch::Tensor avg = (_inputBounds.lower() + _inputBounds.upper()) / 2.0;
    return avg.unsqueeze(0);
}

void NLR::BoundedInputNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias
) {
    (void)inputBounds;

    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

    lbias = torch::Tensor();
    ubias = torch::Tensor();
}

BoundedTensor<torch::Tensor> NLR::BoundedInputNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    (void)inputBounds;
    return _inputBounds;
}

void NLR::BoundedInputNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedInputNode::setOutputSize(unsigned size) {
    _output_size = size;
}

void NLR::BoundedInputNode::setInputBounds(const BoundedTensor<torch::Tensor>& bounds) {
    _inputBounds = BoundedTensor<torch::Tensor>(
        bounds.lower().to(_device),
        bounds.upper().to(_device));
}

BoundedTensor<torch::Tensor> NLR::BoundedInputNode::getInputBounds() const {
    return _inputBounds;
}

void NLR::BoundedInputNode::moveToDevice(const torch::Device& device) {
    BoundedTorchNode::moveToDevice(device);
    _inputBounds = BoundedTensor<torch::Tensor>(
        _inputBounds.lower().to(device),
        _inputBounds.upper().to(device));
}
