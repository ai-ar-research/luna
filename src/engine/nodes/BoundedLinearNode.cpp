// Marabou/src/nlr/bounded_modules/BoundedLinearNode.cpp
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

    // Linear layers don't require input bounds for relaxation
    // _requiresInputBounds is already empty by default (no inputs need bounds)
    // Linear needs CROWN for tight bounds (not IBP), so _ibpIntermediate = false (default)

    // Try to set sizes from weight matrix during construction
    if (_linearModule && _linearModule->weight.defined()) {
        auto weight = _linearModule->weight;
        setInputSize(weight.size(1));  // Weight matrix columns
        setOutputSize(weight.size(0)); // Weight matrix rows

        // Fix weight tensor properties for Alpha-CROWN compatibility
        bool weightNeedsFix = false;
        if (!weight.requires_grad() || !weight.is_contiguous() || weight.dtype() != torch::kFloat32) {
            weightNeedsFix = true;
            if (!weight.requires_grad()) {
            }
            if (!weight.is_contiguous()) {
            }
            if (weight.dtype() != torch::kFloat32) {
            }
        }
        
        // Automatically fix weight tensor: convert to Float32, make contiguous, and enable gradients
        // NOTE: Do NOT use detach() as it breaks the computation graph needed for Alpha-CROWN
        if (weightNeedsFix) {
            _linearModule->weight = weight.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network weights are constants;
        }

        // Similar checks and fixes for bias if it exists
        if (_linearModule->bias.defined()) {
            auto bias = _linearModule->bias;
            bool biasNeedsFix = false;
            if (!bias.requires_grad() || !bias.is_contiguous() || bias.dtype() != torch::kFloat32) {
                biasNeedsFix = true;
                if (!bias.requires_grad()) {
                }
                if (!bias.is_contiguous()) {
                }
                if (bias.dtype() != torch::kFloat32) {
                }
            }
            
            // Automatically fix bias tensor
            // NOTE: Do NOT use detach() as it breaks the computation graph needed for Alpha-CROWN
            if (biasNeedsFix) {
                _linearModule->bias = bias.contiguous().to(torch::kFloat32).requires_grad_(false);  // Network biases are constants;
            }
        }
    }
}

// Forward pass through the linear layer
torch::Tensor BoundedLinearNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically if needed
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = _linearModule->weight.size(0);
    }

    const auto device = input.device();
    torch::Tensor inputFloat = input.to(torch::TensorOptions().dtype(torch::kFloat32).device(device)).contiguous();

    // Use cached weight/bias (alpha already baked into _cached_weight)
    ensureWeightsOnDevice(device);

    // y = (alpha * W) * x + bias
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
    // Invalidate cache - will be repopulated on next use
    _cached_weight = torch::Tensor();
    _cached_bias = torch::Tensor();
}

void BoundedLinearNode::ensureWeightsOnDevice(const torch::Device& device) const
{
    // Fast path: already cached on the right device
    if (_cached_weight.defined() && _cached_device == device) {
        return;
    }

    // Cache weight and bias on target device as float32.
    // _alpha is baked into the cached weight since it's a constant set at construction
    // and never modified. This avoids a scalar multiply on every use.
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

    // Use cached weight/bias on the same device as A (alpha already baked into _cached_weight)
    const auto device = last_lA_tensor.device();
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;
    const auto& bias = _cached_bias;
    
    // For linear layers, A matrices are computed as: A = last_A @ weight
    // where last_A represents the transformation from final output to current layer input
    // and weight represents the transformation from current layer input to current layer output
    //
    // A matrices can have shape:
    //   - 2D: [spec, features] or [features] (legacy format)
    //   - 3D: [spec, batch, features] (standard format from C matrix)
    // Weight has shape: [output_features, input_features]
    //
    // For 3D A: [spec, batch, output_features] @ [output_features, input_features] -> [spec, batch, input_features]
    // For 2D A: [spec, output_features] @ [output_features, input_features] -> [spec, input_features]
    
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
            // torch::matmul handles 3D×2D broadcasting natively:
            // [spec, batch, out_features] @ [out_features, in_features] -> [spec, batch, in_features]
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
    
    // Compute bias contribution: A @ bias = (A * bias).sum(-1) via broadcasting
    // Works for any A shape: [features], [spec, features], [spec, batch, features]
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

    // Set input size from input bounds if not already set
    if (_input_size == 0 && inputLowerBound.defined()) {
        setInputSize(inputLowerBound.numel());
    }

    // Use cached weight/bias (alpha already baked in)
    ensureWeightsOnDevice(device);

    // Compute IBP bounds: y = (alpha * W) * x + bias
    torch::Tensor lowerBound = computeLinearIBPLowerBound(inputLowerBound, inputUpperBound);
    torch::Tensor upperBound = computeLinearIBPUpperBound(inputLowerBound, inputUpperBound);

    // Add bias if defined
    if (_cached_bias.defined()) {
        lowerBound = lowerBound + _cached_bias;
        upperBound = upperBound + _cached_bias;
    }
    
    // Set output size from computed bounds if not already set
    if (_output_size == 0 && lowerBound.defined()) {
        setOutputSize(lowerBound.numel());
    }
    
    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Node information
unsigned BoundedLinearNode::getInputSize() const {
    if (_input_size > 0) {
        return _input_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(1); // Input size is weight matrix columns
    }
    
    return 0;
}

unsigned BoundedLinearNode::getOutputSize() const {
    if (_output_size > 0) {
        return _output_size;
    }
    
    // Fallback: try to infer from weight matrix
    if (_linearModule && _linearModule->weight.defined()) {
        return _linearModule->weight.size(0); // Output size is weight matrix rows
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

// IBP computation methods
torch::Tensor BoundedLinearNode::computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    const auto device = inputLowerBound.device();
    // Use cached weight (alpha already baked in, already on correct device)
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;

    // y_lower = W_positive * x_lower + W_negative * x_upper
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);

    torch::Tensor positive_contribution = torch::matmul(inputLowerBound, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputUpperBound, W_negative.t());

    return positive_contribution + negative_contribution;
}

torch::Tensor BoundedLinearNode::computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound) {
    const auto device = inputLowerBound.device();
    // Use cached weight (alpha already baked in, already on correct device)
    ensureWeightsOnDevice(device);
    const auto& weight = _cached_weight;

    // y_upper = W_positive * x_upper + W_negative * x_lower
    torch::Tensor W_positive = torch::clamp(weight, 0);
    torch::Tensor W_negative = torch::clamp(weight, std::numeric_limits<float>::lowest(), 0);

    torch::Tensor positive_contribution = torch::matmul(inputUpperBound, W_positive.t());
    torch::Tensor negative_contribution = torch::matmul(inputLowerBound, W_negative.t());

    return positive_contribution + negative_contribution;
}

} // namespace NLR
