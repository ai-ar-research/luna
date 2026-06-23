/*********************                                                        */
/*! \file BoundedConstantNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedConstantNode.h"

NLR::BoundedConstantNode::BoundedConstantNode(const torch::Tensor& constantValue, const String& name)
    : _constantValue(constantValue) {
    _nodeName = name;
    _nodeIndex = 0;
}

torch::Tensor NLR::BoundedConstantNode::forward(const torch::Tensor& input) {
    (void)input;
    return _constantValue;
}

void NLR::BoundedConstantNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias
) {
    (void)inputBounds;

    if (last_lA.isPatches() || last_uA.isPatches()) {
        throw std::runtime_error("BoundedConstantNode: Patches mode not implemented (requires conversion)");
    }
    
    torch::Tensor last_lA_tensor = last_lA.asTensor();
    torch::Tensor last_uA_tensor = last_uA.asTensor();

    int spec_size = 0;
    if (last_lA_tensor.defined()) {
        spec_size = (last_lA_tensor.dim() >= 2) ? last_lA_tensor.size(1) : last_lA_tensor.size(0);
    } else if (last_uA_tensor.defined()) {
        spec_size = (last_uA_tensor.dim() >= 2) ? last_uA_tensor.size(1) : last_uA_tensor.size(0);
    }

    auto options = last_lA_tensor.defined() ? last_lA_tensor.options()
                                            : (last_uA_tensor.defined() ? last_uA_tensor.options()
                                                                        : _constantValue.options());
    if (!lbias.defined() && spec_size > 0) {
        lbias = torch::zeros({spec_size}, options);
    }
    if (!ubias.defined() && spec_size > 0) {
        ubias = torch::zeros({spec_size}, options);
    }

    if (last_lA_tensor.defined()) {
        torch::Tensor A_flat;
        if (last_lA_tensor.dim() == 3) {
            A_flat = last_lA_tensor.reshape({last_lA_tensor.size(0) * last_lA_tensor.size(1), last_lA_tensor.size(2)});
        } else if (last_lA_tensor.dim() == 2) {
            A_flat = last_lA_tensor;
        } else {
            A_flat = last_lA_tensor.flatten(0, -2);
        }

        torch::Tensor constant_flat = _constantValue.flatten();

        torch::Tensor new_lbias = torch::matmul(A_flat, constant_flat);

        if (last_lA_tensor.dim() == 3) {
            new_lbias = new_lbias.reshape({last_lA_tensor.size(0), last_lA_tensor.size(1)}).squeeze(0);
        }

        lbias = lbias + new_lbias;
    }

    if (last_uA_tensor.defined()) {
        torch::Tensor A_flat;
        if (last_uA_tensor.dim() == 3) {
            A_flat = last_uA_tensor.reshape({last_uA_tensor.size(0) * last_uA_tensor.size(1), last_uA_tensor.size(2)});
        } else if (last_uA_tensor.dim() == 2) {
            A_flat = last_uA_tensor;
        } else {
            A_flat = last_uA_tensor.flatten(0, -2);
        }

        torch::Tensor constant_flat = _constantValue.flatten();
        torch::Tensor new_ubias = torch::matmul(A_flat, constant_flat);

        if (last_uA_tensor.dim() == 3) {
            new_ubias = new_ubias.reshape({last_uA_tensor.size(0), last_uA_tensor.size(1)}).squeeze(0);
        }

        ubias = ubias + new_ubias;
    }

    outputA_matrices.clear();
}

BoundedTensor<torch::Tensor> NLR::BoundedConstantNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    (void)inputBounds;
    return BoundedTensor<torch::Tensor>(_constantValue, _constantValue);
}

void NLR::BoundedConstantNode::moveToDevice(const torch::Device& device) {
    BoundedTorchNode::moveToDevice(device);
    _constantValue = _constantValue.to(device);
}

void NLR::BoundedConstantNode::setInputSize(unsigned size) {
    (void)size;
}

void NLR::BoundedConstantNode::setOutputSize(unsigned size) {
    (void)size;
}