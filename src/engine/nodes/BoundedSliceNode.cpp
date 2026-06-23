/*********************                                                        */
/*! \file BoundedSliceNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedSliceNode.h"
#include <torch/torch.h>
#include <algorithm>
#include <cmath>

namespace NLR {

constexpr int64_t ONNX_NEG_INF = -9223372036854775807LL;

BoundedSliceNode::BoundedSliceNode(int64_t start, int64_t end, int64_t axes, int64_t steps,
                                   const String& name)
    : _start(start), _end(end), _axes(axes), _steps(steps),
      _nodeName(name), _nodeIndex(0), _input_size(0), _output_size(0) {
}

std::pair<int64_t, int64_t> BoundedSliceNode::_fixup_params(
    const std::vector<int64_t>& shape, int64_t start, int64_t end,
    int64_t axes, int64_t steps) const {

    if (start < 0) {
        start += shape[axes];
    }

    if (end < 0) {
        if (end == ONNX_NEG_INF) {
            // -inf in ONNX: only possible when step == -1
            end = 0;
        } else {
            end += shape[axes];
        }
    }

    if (steps == -1) {
        std::swap(start, end);
        end = end + 1;
    }

    end = std::min(end, shape[axes]);
    start = std::max(start, static_cast<int64_t>(0));

    return {start, end};
}

torch::Tensor BoundedSliceNode::forward(const torch::Tensor& input) {
    _input_shape.clear();
    for (int64_t i = 0; i < input.dim(); ++i) {
        _input_shape.push_back(input.size(i));
    }

    // ONNX axis may include batch dim; adjust when batch dim is implicit
    int64_t effective_axes = _axes;
    if (_axes > 0 && _axes >= input.dim()) {
        effective_axes = _axes - 1;
    }
    if (effective_axes < 0) {
        effective_axes += input.dim();
    }

    std::vector<int64_t> shape_for_fixup = _input_shape;
    auto [fixed_start, fixed_end] = _fixup_params(shape_for_fixup, _start, _end, effective_axes, _steps);

    int64_t length = fixed_end - fixed_start;
    if (length <= 0) {
        std::vector<int64_t> output_shape = _input_shape;
        output_shape[effective_axes] = 0;
        return torch::zeros(output_shape, input.options());
    }

    torch::Tensor result = torch::narrow(input, static_cast<int>(effective_axes),
                                         fixed_start, length);

    if (_steps == -1) {
        result = torch::flip(result, {static_cast<int>(effective_axes)});
    }

    return result;
}

torch::Tensor BoundedSliceNode::forward(const std::vector<torch::Tensor>& inputs) {
    // ONNX start/end/axes/steps handled at parse time; only data input used here
    if (inputs.empty()) {
        throw std::runtime_error("BoundedSliceNode::forward - no inputs provided");
    }
    return forward(inputs[0]);
}

BoundedTensor<torch::Tensor> BoundedSliceNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.empty()) {
        throw std::runtime_error("BoundedSliceNode: no inputs for IBP");
    }

    const auto& dataBounds = inputBounds[0];

    torch::Tensor sliced_lower = forward(dataBounds.lower());
    torch::Tensor sliced_upper = forward(dataBounds.upper());

    return BoundedTensor<torch::Tensor>(sliced_lower, sliced_upper);
}

void BoundedSliceNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (_input_shape.empty() && !inputBounds.empty()) {
        const auto& dataBounds = inputBounds[0];
        torch::Tensor lower = dataBounds.lower();
        for (int64_t i = 0; i < lower.dim(); ++i) {
            _input_shape.push_back(lower.size(i));
        }
    }

    if (_input_shape.empty()) {
        throw std::runtime_error("BoundedSliceNode: input shape not available for backward pass");
    }

    // ONNX axis may include batch dim; adjust when batch dim is implicit
    int64_t effective_axes = _axes;
    int64_t input_ndim = static_cast<int64_t>(_input_shape.size());
    if (_axes > 0 && _axes >= input_ndim) {
        effective_axes = _axes - 1;
    }
    if (effective_axes < 0) {
        effective_axes += input_ndim;
    }

    std::pair<int64_t, int64_t> fixed_params = _fixup_params(_input_shape, _start, _end, effective_axes, _steps);
    int64_t fixed_start = fixed_params.first;
    int64_t fixed_end = fixed_params.second;

    auto _bound_oneside = [this, fixed_start, fixed_end, effective_axes](const BoundA& A, const char* /*name*/) -> BoundA {
        if (!A.defined() || !A.isTensor()) {
            return BoundA();
        }

        torch::Tensor A_tensor = A.asTensor();

        std::vector<int64_t> new_A_shape;

        if (A_tensor.dim() >= 2) {
            new_A_shape.push_back(A_tensor.size(0));
            new_A_shape.push_back(A_tensor.size(1));
        } else if (A_tensor.dim() == 1) {
            new_A_shape.push_back(A_tensor.size(0));
        }

        for (size_t i = 0; i < _input_shape.size(); ++i) {
            new_A_shape.push_back(_input_shape[i]);
        }

        torch::Tensor new_A = torch::zeros(new_A_shape, A_tensor.options());

        // Slice axis in A: offset by leading [spec, batch] dims
        int64_t dim = 2 + effective_axes;
        if (dim < 0) {
            dim += new_A.dim();
        }

        if (dim < 0 || dim >= new_A.dim()) {
            throw std::runtime_error("BoundedSliceNode: invalid dimension for index_copy");
        }

        torch::Tensor indices = torch::arange(fixed_start, fixed_end,
                                               torch::TensorOptions()
                                                   .dtype(torch::kLong)
                                                   .device(A_tensor.device()));

        // Reverse when step == -1: A comes from flipped output
        torch::Tensor source = A_tensor;
        if (_steps == -1) {
            source = torch::flip(A_tensor, {static_cast<int>(dim)});
        }

        new_A = torch::index_copy(new_A, dim, indices, source);

        return BoundA(new_A);
    };

    BoundA lA = _bound_oneside(last_lA, "lA");
    BoundA uA = _bound_oneside(last_uA, "uA");

    outputA_matrices.clear();

    // Only data input gets non-None A; ONNX starts/ends/axes/steps get None
    outputA_matrices.append(Pair<BoundA, BoundA>(lA, uA));

    for (unsigned i = 1; i < 5; ++i) {
        outputA_matrices.append(Pair<BoundA, BoundA>(BoundA(), BoundA()));
    }

    auto options = last_lA.defined() && last_lA.isTensor()
        ? last_lA.asTensor().options()
        : (last_uA.defined() && last_uA.isTensor()
           ? last_uA.asTensor().options()
           : torch::TensorOptions().dtype(torch::kFloat32).device(_device));

    lbias = torch::zeros({1}, options);
    ubias = torch::zeros({1}, options);
}

} // namespace NLR
