#include "BoundedAddNode.h"

namespace NLR {

// Removed writeToCrownTrace - using std::cout for terminal output instead

static inline torch::Tensor firstDefinedBoundTensor(const BoundedTensor<torch::Tensor>& b) {
    if (b.lower().defined()) return b.lower();
    if (b.upper().defined()) return b.upper();
    return torch::Tensor();
}

static inline std::vector<int64_t> boundShapeVec(const BoundedTensor<torch::Tensor>& b) {
    torch::Tensor t = firstDefinedBoundTensor(b);
    if (!t.defined()) return {};
    return t.sizes().vec();
}

static inline bool shapesEqual(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

BoundedAddNode::BoundedAddNode() {
    _input_size = 0;
    _output_size = 0;
    _nodeIndex = 0;
    _nodeName = "add";
}

torch::Tensor BoundedAddNode::forward(const torch::Tensor& input) {
    // Single input forward: add constant if available
    if (_constantValue.defined()) {
        return input + _constantValue;
    }
    return input;
}

void BoundedAddNode::moveToDevice(const torch::Device& device)
{
    BoundedTorchNode::moveToDevice(device);
    if (_constantValue.defined()) {
        _constantValue = _constantValue.to(device);
    }
}

torch::Tensor BoundedAddNode::forward(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() == 1) {
        return forward(inputs[0]);
    } else if (inputs.size() == 2) {
        // Two inputs: x + y
        return inputs[0] + inputs[1];
    } else {
        throw std::runtime_error("BoundedAddNode::forward expects 1 or 2 inputs, got " + std::to_string(inputs.size()));
    }
}

torch::Tensor BoundedAddNode::broadcast_backward(const torch::Tensor& last_A, const BoundedTensor<torch::Tensor>& input) const {
    if (!last_A.defined()) return last_A;


    torch::Tensor x = firstDefinedBoundTensor(input);
    if (!x.defined()) {
        // No shape information to broadcast against. Best-effort: return last_A unchanged.
        return last_A;
    }

    std::vector<int64_t> target_shape = x.sizes().vec();

    // Ensure target has a batch dim matching A when needed.
    // Bounds tensors in this codebase are stored WITHOUT a batch dimension (just [features]).
    // When A is 3D [spec, batch, features], the operand should be [batch, features].
    // We must always prepend a batch dimension when the operand rank is too small for A.

    if (last_A.dim() >= 3) {
        int64_t A_batch = last_A.size(1);
        if ((int64_t)target_shape.size() == 0) {
            target_shape = {A_batch};
        }
        // For 3D A [spec, batch, features], the operand needs at least rank 2 [batch, features].
        // If operand is rank 1 [features], always prepend batch dimension.
        // The old heuristic (target_shape[0] != A_batch) fails when A_batch == features.
        if ((int64_t)target_shape.size() < 2) {
            target_shape.insert(target_shape.begin(), A_batch);
        } else if (target_shape.size() >= 1 && target_shape[0] != A_batch) {
            target_shape.insert(target_shape.begin(), A_batch);
        }
    } else if (last_A.dim() == 2) {
        int64_t A_batch = 1; // 2D A: [spec, features], batch=1 implied
        if ((int64_t)target_shape.size() == 0) {
            target_shape = {A_batch};
        } else if (target_shape.size() >= 1 && target_shape[0] != A_batch) {
            target_shape.insert(target_shape.begin(), A_batch);
        }
    }

    torch::Tensor A = last_A;
    if (A.dim() == 1) {
        // Degenerate; cannot infer batch/spec. Return as-is.
        return A;
    }

    // Determine operand shape including batch dim.
    std::vector<int64_t> operand_shape = target_shape;
    const int64_t op_rank = (int64_t)operand_shape.size();

    // Decide whether A includes a spec dim at position 1.
    // If A.rank == operand_rank + 1, treat dim0 as spec and dim1 as batch; else if equal, treat as elementwise.
    int64_t spec_offset = -1;
    if (A.dim() == op_rank + 1 && A.dim() >= 3) {
        // A has shape [spec, batch, ...] where spec is at dim 0, batch at dim 1
        spec_offset = 1; // [S, B, ...] format (spec at 0, batch at 1)
    } else if (A.dim() == op_rank && A.dim() >= 2) {
        // A has shape [batch, ...] - no spec dimension
        spec_offset = 0; // [B, ...]
    } else if (A.dim() == 2) {
        // Interpret as [spec, features] -> add batch dim 1 and treat as [S,B,features]
        // [spec, features] -> [spec, batch=1, features]
        A = A.unsqueeze(1); // Insert batch dimension at position 1
        spec_offset = 1;
    } else if (A.dim() >= 3) {
        // Fallback: assume [S,B,...] format (spec at 0, batch at 1) if dim>=3.
        spec_offset = 1;
    } else {
        return A;
    }

    if (spec_offset == 1 && A.dim() < 3) return A;

    // Reduce extra dims in A if needed (dims that exist in A but not in operand).
    // Matrix-like: payload begins at dim=2; elementwise: payload begins at dim=1.
    const int64_t payload_start = (spec_offset == 1) ? 2 : 1;
    const int64_t A_payload_dims = A.dim() - payload_start;
    const int64_t op_payload_dims = op_rank - 1; // excluding batch
    if (A_payload_dims > op_payload_dims) {
        // Special case: operand bounds are flattened (e.g. [flat] or [B, flat]) but represent
        // the same underlying tensor as A's payload (e.g. [C,H,W]). In this case, DO NOT
        // reduce/sum A's extra dimensions. This was previously collapsing A from
        // [B,S,C,H,W] -> [B,S,W] (or similar), which is incorrect for residual adds.
        int64_t op_payload_numel = 1;
        for (int64_t i = 1; i < op_rank; ++i) op_payload_numel *= operand_shape[(size_t)i];
        int64_t A_payload_numel = 1;
        for (int64_t i = payload_start; i < A.dim(); ++i) A_payload_numel *= A.size(i);

        if (op_payload_dims == 1 && op_payload_numel == A_payload_numel) {
            // Keep A as-is.
        } else {
            const int64_t num_extra = A_payload_dims - op_payload_dims;
            std::vector<int64_t> sum_dims;
            sum_dims.reserve((size_t)num_extra);
            for (int64_t i = 0; i < num_extra; ++i) {
                sum_dims.push_back(payload_start + i);
            }
            if (!sum_dims.empty()) {
                A = A.sum(sum_dims);
            }
        }
    }

    // Sum-reduce broadcasted dims (keepdim) where operand dim == 1 but A dim != 1.
    // IMPORTANT: A format is [spec, batch, features], so:
    // - spec_offset == 1 means: spec at dim 0, batch at dim 1, features at dim 2+
    // - For operand shape [batch, ...], we need to map operand dim i to A dim (i+1) when spec_offset==1
    std::vector<int64_t> keep_dims;
    if (op_rank >= 2) {
        for (int64_t i = 1; i < op_rank; ++i) {
            int64_t op_dim = operand_shape[(size_t)i];
            // Map operand dimension i to A dimension
            // Operand: [batch, ...] where batch is at dim 0, payload starts at dim 1
            // A: [spec, batch, features] where spec is at dim 0, batch at dim 1, features at dim 2+
            // So operand dim i maps to A dim (i+1) when spec_offset==1
            int64_t a_dim_index = (spec_offset == 1) ? (i + 1) : i;
            if (a_dim_index < 0 || a_dim_index >= A.dim()) continue;
            int64_t A_dim = A.size(a_dim_index);
            if (op_dim == 1 && A_dim != 1) {
                keep_dims.push_back(a_dim_index);
            }
        }
    }
    if (!keep_dims.empty()) {
        A = A.sum(keep_dims, /*keepdim=*/true);
    }

    return A;
}

void BoundedAddNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    // For Add node: the backward bound propagation simply passes through the A matrices
    // Since d(x+y)/dx = 1 and d(x+y)/dy = 1

    // Clear output matrices
    outputA_matrices.clear();

    auto bound_one_side = [&](const BoundA& last_A, const BoundedTensor<torch::Tensor>& in) -> BoundA {
        if (!last_A.defined()) return BoundA();
        if (last_A.isTensor()) {
            torch::Tensor A = broadcast_backward(last_A.asTensor(), in);
            return BoundA(A);
        } else {
            // Patches: broadcasting is not supported here; handled at the call sites.
            return last_A;
        }
    };

    if (inputBounds.size() == 1) {
        // Single input case (x + constant)
        // The gradient w.r.t. x is 1, so we just pass through the A matrices
        BoundA lA_x = bound_one_side(last_lA, inputBounds[0]);
        BoundA uA_x = bound_one_side(last_uA, inputBounds[0]);
        outputA_matrices.append(Pair<BoundA, BoundA>(lA_x, uA_x));

        // Add contribution from constant to bias if we have one
        if (_constantValue.defined()) {
            if (last_lA.isPatches() || last_uA.isPatches()) {
                throw std::runtime_error("BoundedAddNode: Patches mode with constant bias not implemented (requires conversion)");
            }

            torch::Tensor lA_tensor = last_lA.asTensor();
            torch::Tensor uA_tensor = last_uA.asTensor();

            // For x + c, we need to compute A @ c where A is the backward propagation matrix
            // This properly accounts for the linear transformation in the backward pass
            if (lA_tensor.defined()) {
                // Broadcast constant to the input shape so bias computation matches broadcast semantics.
                torch::Tensor x = firstDefinedBoundTensor(inputBounds[0]);
                torch::Tensor c_full = x.defined() ? _constantValue.to(x.device()).to(x.dtype()) : _constantValue;
                if (x.defined()) c_full = c_full.expand_as(x);
                // Generic bias contribution: sum(A * c_full) over elementwise dimensions.
                torch::Tensor constant_contrib;
                if (lA_tensor.dim() == c_full.dim()) {
                    // Elementwise A: [B,...] -> reduce over dims [1..]
                    std::vector<int64_t> sum_dims;
                    for (int64_t d = 1; d < lA_tensor.dim(); ++d) sum_dims.push_back(d);
                    constant_contrib = (lA_tensor * c_full).sum(sum_dims);
                } else if (lA_tensor.dim() == c_full.dim() + 1) {
                    // Matrix-like A: [spec, batch, ...] -> reduce over dims [2..]
                    // A format: [spec, batch, features], c_full: [features]
                    // After unsqueeze(1): c_full becomes [1, features] or [1, 1, features] depending on c_full.dim()
                    // Multiply and sum over feature dims [2..] -> [spec, batch]
                    std::vector<int64_t> sum_dims;
                    for (int64_t d = 2; d < lA_tensor.dim(); ++d) sum_dims.push_back(d);
                    constant_contrib = (lA_tensor * c_full.unsqueeze(1)).sum(sum_dims);
                    // Keep as [spec, batch] format - don't squeeze spec dimension even if spec=1
                    // This matches Python behavior where bias is always [spec, batch]
                } else if (lA_tensor.dim() == 2) {
                    // [S,features] special-case
                    torch::Tensor constant = c_full.flatten();
                    constant_contrib = torch::matmul(lA_tensor, constant);
                } else if (lA_tensor.dim() == 3) {
                    // [spec, batch, features] special-case
                    // A format: [spec, batch, features], constant: [features]
                    // matmul: [spec, batch, features] @ [features, 1] -> [spec, batch, 1]
                    // squeeze(-1): [spec, batch, 1] -> [spec, batch]
                    torch::Tensor constant = c_full.flatten();
                    constant_contrib = torch::matmul(lA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                    // Keep as [spec, batch] format - don't squeeze spec dimension even if spec=1
                    // This matches Python behavior where bias is always [spec, batch]
                } else {
                    throw std::runtime_error("BoundedAddNode::boundBackward: unsupported last_lA shape for constant bias");
                }

                if (lbias.defined()) {
                    lbias = lbias + constant_contrib;
                } else {
                    lbias = constant_contrib;
                }
            }

            if (uA_tensor.defined()) {
                torch::Tensor x = firstDefinedBoundTensor(inputBounds[0]);
                torch::Tensor c_full = x.defined() ? _constantValue.to(x.device()).to(x.dtype()) : _constantValue;
                if (x.defined()) c_full = c_full.expand_as(x);
                torch::Tensor constant_contrib;
                if (uA_tensor.dim() == c_full.dim()) {
                    std::vector<int64_t> sum_dims;
                    for (int64_t d = 1; d < uA_tensor.dim(); ++d) sum_dims.push_back(d);
                    constant_contrib = (uA_tensor * c_full).sum(sum_dims);
                } else if (uA_tensor.dim() == c_full.dim() + 1) {
                    // Matrix-like A: [spec, batch, ...] -> reduce over dims [2..]
                    // A format: [spec, batch, features], c_full: [features]
                    // After unsqueeze(1): c_full becomes [1, features] or [1, 1, features] depending on c_full.dim()
                    // Multiply and sum over feature dims [2..] -> [spec, batch]
                    std::vector<int64_t> sum_dims;
                    for (int64_t d = 2; d < uA_tensor.dim(); ++d) sum_dims.push_back(d);
                    constant_contrib = (uA_tensor * c_full.unsqueeze(1)).sum(sum_dims);
                    // Keep as [spec, batch] format - don't squeeze spec dimension even if spec=1
                    // This matches Python behavior where bias is always [spec, batch]
                } else if (uA_tensor.dim() == 2) {
                    torch::Tensor constant = c_full.flatten();
                    constant_contrib = torch::matmul(uA_tensor, constant);
                } else if (uA_tensor.dim() == 3) {
                    // [spec, batch, features] special-case
                    // A format: [spec, batch, features], constant: [features]
                    // matmul: [spec, batch, features] @ [features, 1] -> [spec, batch, 1]
                    // squeeze(-1): [spec, batch, 1] -> [spec, batch]
                    torch::Tensor constant = c_full.flatten();
                    constant_contrib = torch::matmul(uA_tensor, constant.unsqueeze(-1)).squeeze(-1);
                    // Keep as [spec, batch] format - don't squeeze spec dimension even if spec=1
                    // This matches Python behavior where bias is always [spec, batch]
                } else {
                    throw std::runtime_error("BoundedAddNode::boundBackward: unsupported last_uA shape for constant bias");
                }

                if (ubias.defined()) {
                    ubias = ubias + constant_contrib;
                } else {
                    ubias = constant_contrib;
                }
            }
        }
    } else if (inputBounds.size() == 2) {
        // Two input case (x + y)
        // Patches mode: we only support same-shape residual adds.
        // If shapes are different (broadcasting), throw (not supported for patches).
        const bool anyPatches = (last_lA.defined() && last_lA.isPatches()) || (last_uA.defined() && last_uA.isPatches());
        if (anyPatches) {
            auto sx = boundShapeVec(inputBounds[0]);
            auto sy = boundShapeVec(inputBounds[1]);
            if (!sx.empty() && !sy.empty() && !shapesEqual(sx, sy)) {
                throw std::runtime_error("BoundedAddNode: Patches mode Add requires identical input shapes (broadcasting not supported)");
            }
        }

        // Both inputs get the same A matrices since derivatives are 1,
        // but each input may require broadcast-aware reduction (AutoLiRPA behavior).
        BoundA lA_x = bound_one_side(last_lA, inputBounds[0]);
        BoundA uA_x = bound_one_side(last_uA, inputBounds[0]);
        BoundA lA_y = bound_one_side(last_lA, inputBounds[1]);
        BoundA uA_y = bound_one_side(last_uA, inputBounds[1]);
        outputA_matrices.append(Pair<BoundA, BoundA>(lA_x, uA_x));
        outputA_matrices.append(Pair<BoundA, BoundA>(lA_y, uA_y));

        // IMPORTANT: Following auto_LiRPA, BoundAdd returns zero bias for Add(x,y)
        // The bias terms are already initialized to undefined/zero by the caller
    } else {
        throw std::runtime_error("BoundedAddNode::boundBackward expects 1 or 2 input bounds");
    }
}

BoundedTensor<torch::Tensor> BoundedAddNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {

    if (inputBounds.size() == 1) {
        // Single input case (x + constant)
        const auto& x = inputBounds[0];
        if (_constantValue.defined()) {
            // Add constant to both lower and upper bounds
            torch::Tensor lower = x.lower() + _constantValue;
            torch::Tensor upper = x.upper() + _constantValue;
            return BoundedTensor<torch::Tensor>(lower, upper);
        } else {
            // No constant, just pass through
            return x;
        }
    } else if (inputBounds.size() == 2) {
        // Two input case (x + y)
        const auto& x = inputBounds[0];
        const auto& y = inputBounds[1];

        // For addition: [a,b] + [c,d] = [a+c, b+d]
        torch::Tensor lower = x.lower() + y.lower();
        torch::Tensor upper = x.upper() + y.upper();

        return BoundedTensor<torch::Tensor>(lower, upper);
    } else {
        throw std::runtime_error("BoundedAddNode::computeIntervalBoundPropagation expects 1 or 2 input bounds");
    }
}

void BoundedAddNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedAddNode::setOutputSize(unsigned size) {
    _output_size = size;
}

} // namespace NLR
