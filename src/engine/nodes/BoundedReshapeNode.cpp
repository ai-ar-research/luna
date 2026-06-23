/*********************                                                        */
/*! \file BoundedReshapeNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedReshapeNode.h"

namespace NLR {

BoundedReshapeNode::BoundedReshapeNode(const Operations::ReshapeWrapper& reshape_module)
    : _reshape_module(reshape_module) {
    _nodeName = "reshape";
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
}

torch::Tensor BoundedReshapeNode::forward(const torch::Tensor& input) {
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }

    _input_shape.clear();
    for (int i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }

    torch::Tensor output = _reshape_module.forward(input);

    return output;
}

void BoundedReshapeNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReshapeNode expects at least one input");
    }

    if (last_lA.isPatches() || last_uA.isPatches()) {
         throw std::runtime_error("BoundedReshapeNode: Patches propagation not implemented (convert to matrix)");
    }

    torch::Tensor lA = last_lA.asTensor();
    torch::Tensor uA = last_uA.asTensor();

    auto _bound_oneside = [&](const torch::Tensor& A) -> torch::Tensor {
        if (!A.defined()) {
            return torch::Tensor();
        }

        if (_input_shape.empty()) {
            return A;
        }

        if (A.dim() < 2) return A;

        std::vector<int64_t> new_shape;
        new_shape.push_back(A.size(0));
        new_shape.push_back(A.size(1));

        // Skip batch dim at index 0 in _input_shape
        for (size_t i = 1; i < _input_shape.size(); ++i) {
            new_shape.push_back(_input_shape[i]);
        }

        return A.reshape(new_shape);
    };

    torch::Tensor reshaped_lA = _bound_oneside(lA);
    torch::Tensor reshaped_uA = _bound_oneside(uA);

    outputA_matrices.clear();
    outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(reshaped_lA), BoundA(reshaped_uA)));

    if (lA.defined()) {
        int output_size = lA.size(1);

        if (!lbias.defined()) {
            lbias = torch::zeros({output_size}, lA.options());
        }
    } else {
        if (!lbias.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
            lbias = torch::zeros({1}, options);
        }
    }

    if (uA.defined()) {
        int output_size = uA.size(1);

        if (!ubias.defined()) {
            ubias = torch::zeros({output_size}, uA.options());
        }
    } else {
        if (!ubias.defined()) {
            auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
            ubias = torch::zeros({1}, options);
        }
    }
}

BoundedTensor<torch::Tensor> BoundedReshapeNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Reshape module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();

    torch::Tensor reshapedLower = _reshape_module.forward(inputLowerBound);
    torch::Tensor reshapedUpper = _reshape_module.forward(inputUpperBound);

    return BoundedTensor<torch::Tensor>(reshapedLower, reshapedUpper);
}

void NLR::BoundedReshapeNode::setInputSize(unsigned size) {
    _input_size = size;
}

void NLR::BoundedReshapeNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
