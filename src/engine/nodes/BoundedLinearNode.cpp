/*********************                                                        */
/*! \file BoundedLinearNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedLinearNode.h"
#include <sstream>

namespace NLR {

NLR::BoundedLinearNode::BoundedLinearNode(const torch::nn::Linear& linearModule,
    float alpha, const String& name)
    : _linearModule(linearModule),
      _alpha(alpha) {

    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    if (_linearModule && _linearModule->weight.defined()) {
        auto weight = _linearModule->weight;
        setInputSize(weight.size(1));
        setOutputSize(weight.size(0));

        bool weightNeedsFix = false;
        if (!weight.requires_grad() || !weight.is_contiguous() || weight.dtype() != torch::kFloat32) {
            weightNeedsFix = true;
        }

        // Do NOT use detach() - breaks computation graph needed for Alpha-CROWN
        if (weightNeedsFix) {
            _linearModule->weight = weight.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network weights are constants
        }

        if (_linearModule->bias.defined()) {
            auto bias = _linearModule->bias;
            bool biasNeedsFix = false;
            if (!bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32) {
                biasNeedsFix = true;
            }

            if (biasNeedsFix) {
                _linearModule->bias = bias.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network biases are constants
            }
        }
    }
}

torch::Tensor BoundedLinearNode::forward(const torch::Tensor& input) {
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = _linearModule->weight.size(0);
    }

    const auto device = input.device();
    torch::Tensor inputFloat = input.to(torch::TensorOptions().dtype(torch::kFloat32).device(device)).contiguous();

    // alpha already baked into _cached_weight
    ensureWeightsOnDevice(device);

    torch::Tensor result = torch::matmul(inputFloat, _cached_weight.t());

    if (_cached_bias.defined()) {
        result = result + _cached_bias;
    }

    return result;
}

void BoundedLinearNode::moveToDevice(const torch::Device& device)
{
    BoundedTorchNode::moveToDevice(device);
    _linearModule->to(device);
    _cached_weight = torch::Tensor();
    _cached_bias = torch::Tensor();
}

void BoundedLinearNode::ensureWeightsOnDevice(const torch::Device& device) const
{
    if (_cached_weight.defined() && _cached_device == device) {
        return;
    }

    // _alpha baked in as constant - avoids scalar multiply on every use
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    _cached_weight = (_alpha * _linearModule->weight).to(options).contiguous();
    if (_linearModule->bias.defined()) {
        _cached_bias = _linearModule->bias.to(options);
    } else {
        _cached_bias = torch::Tensor();
    }
    _cached_device = device;
}

void BoundedLinearNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedLinearNode expects at least one input");
    }

    if (!last_lA.defined() || !last_uA.defined()) {
        throw std::runtime_error("BoundedLinearNode: last_lA and last_uA must be defined");
    }
    if (!last_lA.isTensor() || !last_uA.isTensor()) {
        throw std::runtime_error("BoundedLinearNode: Patches mode propagation not implemented (requires conversion to matrix)");
    }

    torch::Tensor last_lA_tensor = last_lA.asTensor();
    torch::Tensor last_uA_tensor = last_uA.asTensor();

    const auto device = last_lA_tensor.device();
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;
    const auto& bias = _cached_bias;

    auto projectA = [&](const torch::Tensor& A_in) -> torch::Tensor {
        if (!A_in.defined()) return torch::Tensor();
        if (A_in.dim() == 3) {
            long output_features = A_in.size(2);
            if (weight.size(0) != output_features) {
                std::ostringstream oss;
                oss << "BoundedLinearNode::boundBackward: dimension mismatch - A matrix has " << output_features
                    << " features but weight has " << weight.size(0) << " output features";
                throw std::runtime_error(oss.str());
            }
            return torch::matmul(A_in, weight);
        } else if (A_in.dim() == 2) {
            long spec_dim = A_in.size(0);
            long features = A_in.size(1);
            if (weight.size(0) != features) {
                if (weight.size(0) == spec_dim && weight.size(0) != features) {
                    std::ostringstream oss;
                    oss << "BoundedLinearNode::boundBackward: A matrix appears TRANSPOSED!\n"
                        << "  Expected: [spec, features] but got: [" << spec_dim << ", " << features << "]\n"
                        << "  Weight shape: [" << weight.size(0) << ", " << weight.size(1) << "]";
                    throw std::runtime_error(oss.str());
                } else {
                    std::ostringstream oss;
                    oss << "BoundedLinearNode::boundBackward: 2D A matrix dimension mismatch - A has " << features
                        << " features (dim 1) but weight has " << weight.size(0) << " output features. "
                        << "A shape: [" << spec_dim << ", " << features << "], weight shape: ["
                        << weight.size(0) << ", " << weight.size(1) << "]";
                    throw std::runtime_error(oss.str());
                }
            }
            return torch::matmul(A_in, weight);
        } else if (A_in.dim() == 1) {
            return torch::matmul(A_in.unsqueeze(0), weight);
        } else {
            std::ostringstream oss;
            oss << "BoundedLinearNode::boundBackward: unsupported A matrix dimension " << A_in.dim()
                << ", weight shape: [" << weight.size(0) << ", " << weight.size(1) << "]";
            throw std::runtime_error(oss.str());
        }
    };

    torch::Tensor lA = projectA(last_lA_tensor);
    torch::Tensor uA = projectA(last_uA_tensor);

    outputA_matrices.append(Pair<BoundA, BoundA>(
        BoundA(lA), BoundA(uA)));

    if (bias.defined()) {
        if (last_lA_tensor.defined() && last_lA_tensor.numel() > 0) {
            lbias = (last_lA_tensor * bias).sum(-1);
            if (last_lA_tensor.dim() == 2) {
                lbias = lbias.unsqueeze(-1);
            } else if (last_lA_tensor.dim() == 1) {
                lbias = lbias.unsqueeze(0).unsqueeze(-1);
            }
        }
        if (last_uA_tensor.defined() && last_uA_tensor.numel() > 0) {
            ubias = (last_uA_tensor * bias).sum(-1);
            if (last_uA_tensor.dim() == 2) {
                ubias = ubias.unsqueeze(-1);
            } else if (last_uA_tensor.dim() == 1) {
                ubias = ubias.unsqueeze(0).unsqueeze(-1);
            }
        }
    }

}

BoundedTensor<torch::Tensor> BoundedLinearNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("Linear module requires at least one input");
    }

    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower().to(torch::kFloat32);
    torch::Tensor inputUpperBound = inputBoundsPair.upper().to(torch::kFloat32);
    const auto device = inputLowerBound.device();

    if (_input_size == 0 && inputLowerBound.defined()) {
        setInputSize(inputLowerBound.numel());
    }

    ensureWeightsOnDevice(device);

    torch::Tensor lowerBound = computeLinearIBPLowerBound(inputLowerBound, inputUpperBound);
    torch::Tensor upperBound = computeLinearIBPUpperBound(inputLowerBound, inputUpperBound);

    if (_cached_bias.defined()) {
        lowerBound = lowerBound + _cached_bias;
        upperBound = upperBound + _cached_bias;
    }

    if (_output_size == 0 && lowerBound.defined()) {
        setOutputSize(lowerBound.numel());
    }

    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

unsigned BoundedLinearNode::getInputSize() const {
    if (_input_size > 0) {
        return _input_size;
    }

    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(1);
    }

    return 0;
}

unsigned BoundedLinearNode::getOutputSize() const {
    if (_output_size > 0) {
        return _output_size;
    }

    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(0);
    }

    return 0;
}

void BoundedLinearNode::setInputSize(unsigned size) {
    if (size > 0) {
        _input_size = size;
    }
}

void BoundedLinearNode::setOutputSize(unsigned size) {
    if (size > 0) {
        _output_size = size;
    }
}

torch::Tensor BoundedLinearNode::computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    const auto device = inputLowerBound.device();
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;

    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);

    torch::Tensor positive_contribution = torch::matmul(inputLowerBound, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputUpperBound, W_negative.t());

    return positive_contribution + negative_contribution;
}

torch::Tensor BoundedLinearNode::computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    const auto device = inputLowerBound.device();
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;

    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);

    torch::Tensor positive_contribution = torch::matmul(inputUpperBound, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputLowerBound, W_negative.t());

    return positive_contribution + negative_contribution;
}

} // namespace NLR
