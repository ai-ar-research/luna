/*********************                                                        */
/*! \file BoundedSubNode.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "BoundedSubNode.h"

namespace NLR {

BoundedSubNode::BoundedSubNode() {
    _nodeName = "sub";
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;
    _constantIsSecond = true;
}

torch::Tensor BoundedSubNode::forward(const torch::Tensor& input) {
    if (_constantValue.defined()) {
        if (_constantIsSecond) {
            return input - _constantValue;
        } else {
            return _constantValue - input;
        }
    }
    return input;
}

void BoundedSubNode::moveToDevice(const torch::Device& device)
{
    BoundedTorchNode::moveToDevice(device);
    if (_constantValue.defined()) {
        _constantValue = _constantValue.to(device);
    }
}

torch::Tensor BoundedSubNode::forward(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("BoundedSubNode::forward() expects exactly 2 inputs for subtraction (x - y)");
    }

    const torch::Tensor& x = inputs[0];
    const torch::Tensor& y = inputs[1];

    torch::Tensor result = x - y;

    if (x.dim() > 0) {
        _input_size = x.numel();
        _output_size = result.numel();
    }

    return result;
}

void BoundedSubNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    outputA_matrices.clear();

    if (inputBounds.size() == 1) {
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                // x - constant: gradient w.r.t. x is 1
                outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

                if (last_lA.isPatches() || last_uA.isPatches()) {
                    throw std::runtime_error("BoundedSubNode: Patches mode with constant bias not implemented (requires conversion)");
                }

                torch::Tensor last_lA_tensor = last_lA.asTensor();
                torch::Tensor last_uA_tensor = last_uA.asTensor();

                if (last_lA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.to(last_lA_tensor.device()).flatten();

                    torch::Tensor constant_contrib;
                    if (last_lA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_lA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    lbias = -constant_contrib;
                }

                if (last_uA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.to(last_uA_tensor.device()).flatten();

                    torch::Tensor constant_contrib;
                    if (last_uA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) {
                            constant_contrib = constant_contrib.squeeze(0);
                        }
                    } else if (last_uA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    ubias = -constant_contrib;
                }
            } else {
                // constant - x: gradient w.r.t. x is -1
                BoundA neg_lA, neg_uA;

                if (last_lA.isTensor()) {
                    neg_lA = last_lA.asTensor().defined() ? BoundA(-last_lA.asTensor()) : BoundA();
                } else {
                    auto p = last_lA.asPatches();
                    neg_lA = BoundA(p->create_similar(-p->patches));
                }

                if (last_uA.isTensor()) {
                    neg_uA = last_uA.asTensor().defined() ? BoundA(-last_uA.asTensor()) : BoundA();
                } else {
                    auto p = last_uA.asPatches();
                    neg_uA = BoundA(p->create_similar(-p->patches));
                }

                outputA_matrices.append(Pair<BoundA, BoundA>(neg_lA, neg_uA));

                if (last_lA.isPatches() || last_uA.isPatches()) {
                    throw std::runtime_error("BoundedSubNode: Patches mode with constant bias not implemented (requires conversion)");
                }

                torch::Tensor last_lA_tensor = last_lA.asTensor();
                torch::Tensor last_uA_tensor = last_uA.asTensor();

                if (last_lA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.to(last_lA_tensor.device()).flatten();

                    torch::Tensor constant_contrib;
                    if (last_lA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) constant_contrib = constant_contrib.squeeze(0);
                    } else if (last_lA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_lA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_lA dimensions");
                    }

                    lbias = constant_contrib;
                }

                if (last_uA_tensor.defined()) {
                    torch::Tensor constant = _constantValue.to(last_uA_tensor.device()).flatten();

                    torch::Tensor constant_contrib;
                    if (last_uA_tensor.dim() == 3) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                        if (constant_contrib.size(0) == 1) constant_contrib = constant_contrib.squeeze(0);
                    } else if (last_uA_tensor.dim() == 2) {
                        constant_contrib = torch::matmul(last_uA_tensor, constant);
                    } else {
                        throw std::runtime_error("BoundedSubNode::boundBackward: unexpected last_uA dimensions");
                    }

                    ubias = constant_contrib;
                }
            }
        } else {
            throw std::runtime_error("BoundedSubNode::boundBackward with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        outputA_matrices.append(Pair<BoundA, BoundA>(last_lA, last_uA));

        // Second input (y): negative sign, SWAPPED A matrices
        BoundA neg_lA, neg_uA;

        if (last_lA.isTensor()) {
            neg_lA = last_lA.asTensor().defined() ? BoundA(-last_lA.asTensor()) : BoundA();
        } else {
            auto p = last_lA.asPatches();
            neg_lA = BoundA(p->create_similar(-p->patches));
        }

        if (last_uA.isTensor()) {
            neg_uA = last_uA.asTensor().defined() ? BoundA(-last_uA.asTensor()) : BoundA();
        } else {
            auto p = last_uA.asPatches();
            neg_uA = BoundA(p->create_similar(-p->patches));
        }

        // Swap neg_uA and neg_lA for correct bound direction
        outputA_matrices.append(Pair<BoundA, BoundA>(neg_uA, neg_lA));
    } else {
        throw std::runtime_error("BoundedSubNode::boundBackward expects at least 1 input bound");
    }

    if (last_lA.isPatches() || last_uA.isPatches()) {
        // Patches mode: undefined bias implies zero contribution
    } else {
        torch::Tensor last_lA_tensor = last_lA.asTensor();
        torch::Tensor last_uA_tensor = last_uA.asTensor();

        if (last_lA_tensor.defined()) {
            int output_size = last_lA_tensor.size(1);
            if (!lbias.defined()) {
                lbias = torch::zeros({output_size}, last_lA_tensor.options());
            } else if (lbias.numel() != output_size) {
                if (lbias.numel() == 1) lbias = lbias.expand({output_size});
            }
        } else {
            if (!lbias.defined()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
                lbias = torch::zeros({1}, options);
            }
        }

        if (last_uA_tensor.defined()) {
            int output_size = last_uA_tensor.size(1);
            if (!ubias.defined()) {
                ubias = torch::zeros({output_size}, last_uA_tensor.options());
            } else if (ubias.numel() != output_size) {
                if (ubias.numel() == 1) ubias = ubias.expand({output_size});
            }
        } else {
            if (!ubias.defined()) {
                auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_device);
                ubias = torch::zeros({1}, options);
            }
        }
    }
}

BoundedTensor<torch::Tensor> BoundedSubNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() == 1) {
        const auto& xBounds = inputBounds[0];
        if (_constantValue.defined()) {
            if (_constantIsSecond) {
                torch::Tensor resultLower = xBounds.lower() - _constantValue;
                torch::Tensor resultUpper = xBounds.upper() - _constantValue;
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            } else {
                torch::Tensor resultLower = _constantValue - xBounds.upper();
                torch::Tensor resultUpper = _constantValue - xBounds.lower();
                return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
            }
        } else {
            throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation with 1 input requires a constant value");
        }
    } else if (inputBounds.size() >= 2) {
        const auto& xBounds = inputBounds[0];
        const auto& yBounds = inputBounds[1];

        torch::Tensor resultLower = xBounds.lower() - yBounds.upper();
        torch::Tensor resultUpper = xBounds.upper() - yBounds.lower();

        return BoundedTensor<torch::Tensor>(resultLower, resultUpper);
    } else {
        throw std::runtime_error("BoundedSubNode::computeIntervalBoundPropagation requires at least 1 input");
    }
}

void BoundedSubNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedSubNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
