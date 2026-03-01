#include "BoundedReLUNode.h"
#include "AlphaCROWNAnalysis.h"
#include "LunaConfiguration.h"
#include "conv/Patches.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace NLR {

BoundedReLUNode::BoundedReLUNode(const torch::nn::ReLU& reluModule, const String& name)
    : BoundedAlphaOptimizeNode()
    , _reluModule(std::make_shared<torch::nn::ReLU>(reluModule)) {
    _nodeName = name;
    _nodeIndex = 0;
    _input_size = 0;
    _output_size = 0;

    // Lazy computation flags (auto_LiRPA style)
    _requiresInputBounds.append(0);  // ReLU needs bounds on input 0 for relaxation
    _ibpIntermediate = true;          // ReLU can use IBP for intermediate bounds
}

// Forward pass through the ReLU layer
torch::Tensor BoundedReLUNode::forward(const torch::Tensor& input) {
    // Update input/output sizes dynamically
    if (input.dim() > 0) {
        _input_size = input.numel();
        _output_size = input.numel();
    }
    
    // Apply ReLU transformation
    return (*_reluModule)(input);
}

void BoundedReLUNode::moveToDevice(const torch::Device& device)
{
    BoundedAlphaOptimizeNode::moveToDevice(device);
    if (_reluModule) {
        _reluModule->ptr()->to(device);
    }
    if (init_upper_d.defined()) {
        init_upper_d = init_upper_d.to(device);
    }
}

// Auto-LiRPA style boundBackward method
void BoundedReLUNode::boundBackward(
    const BoundA& last_lA,
    const BoundA& last_uA,
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
    Vector<Pair<BoundA, BoundA>>& outputA_matrices,
    torch::Tensor& lbias,
    torch::Tensor& ubias) {

    if (inputBounds.size() < 1) {
        throw std::runtime_error("BoundedReLUNode expects at least one input");
    }

    const auto& inputBound = inputBounds[0];
    torch::Tensor input_lower = inputBound.lower();
    torch::Tensor input_upper = inputBound.upper();

    const BoundA& effective_lA = last_lA;
    const BoundA& effective_uA = last_uA;

    // Extract live spec dimension from last_lA or last_uA for per-spec alpha
    int specDim = 1;
    auto inferSpecDim = [](const BoundA& A) -> int {
        if (!A.defined() || !A.isTensor()) return 1;
        torch::Tensor t = A.asTensor();
        if (!t.defined()) return 1;
        // A matrix conventions in this codebase:
        // - 3D A: [S, B, ...] => spec dim is size(0), batch dim is size(1)
        // - 2D A: [S, ...]    => spec dim is size(0)
        // For alpha-CROWN, the spec dimension from the output backward pass is what matters
        // When A shape is [S, 1, neurons], S is the number of specs we optimize for
        if (t.dim() >= 3) return (int)t.size(0);
        if (t.dim() == 2) return (int)t.size(0);
        return 1;
    };
    if (effective_lA.defined() && effective_lA.isTensor()) {
        specDim = inferSpecDim(effective_lA);
    } else if (effective_uA.defined() && effective_uA.isTensor()) {
        specDim = inferSpecDim(effective_uA);
    }
    _currentSpecDim = specDim;  // Store for use in _maskAlpha

    // DEBUG: Print spec dimension
    if (LunaConfiguration::VERBOSE) {
        printf("[DEBUG BoundedReLUNode::backward] node=%u, specDim=%d", getNodeIndex(), specDim);
        if (effective_lA.defined() && effective_lA.isTensor()) {
            auto t = effective_lA.asTensor();
            printf(", lA.dim=%d, lA.shape=[", (int)t.dim());
            for (int i = 0; i < t.dim(); ++i) {
                if (i > 0) printf(",");
                printf("%lld", (long long)t.size(i));
            }
            printf("]");
        }
        printf("\n");
    }

    // Call the unified backward relaxation method
    auto relaxation_result = _backwardRelaxation(effective_lA, effective_uA, input_lower, input_upper);

    // Helper lambdas
    auto expand_like = [](torch::Tensor v, const torch::Tensor& A) {
        if (!v.defined() || !A.defined()) return v;

        // Want v to broadcast to A.
        // Typical shapes:
        // - v: [flat] and A: [spec, batch, flat] -> v needs [1, 1, flat]
        // - v: [C,H,W] and A: [spec, batch, C, H, W] -> v needs [1, 1, C, H, W]
        // - v: [flat] but A is conv-shaped [spec, batch, C, H, W] (flat == C*H*W)
        // - v: [spec, flat] and A: [spec, batch, flat] -> v needs [spec, 1, flat] (per-spec alpha!)
        //
        // Handle per-spec alpha case first: v=[spec, out], A=[spec, batch, out]
        if (v.dim() == 2 && A.dim() == 3 && v.size(0) == A.size(0)) {
            // Per-spec slopes: [spec, out] -> [spec, 1, out]
            v = v.unsqueeze(1);
            try {
                return v.expand_as(A);
            } catch (...) {
                return v;
            }
        }

        // Handle the common "flat-to-conv" mismatch by reshaping v when its numel matches
        // the product of A's payload dims.
        if (v.numel() > 0 && A.dim() >= 3) {
            int64_t payload_numel = 1;
            for (int d = 2; d < A.dim(); ++d) payload_numel *= A.size(d);
            if (v.numel() == payload_numel && v.dim() == 1) {
                // v: [flat] -> [1,1, *payload]
                std::vector<int64_t> payload_shape;
                for (int d = 2; d < A.dim(); ++d) payload_shape.push_back(A.size(d));
                v = v.view(payload_shape).unsqueeze(0).unsqueeze(0);
            } else if (v.numel() == payload_numel && v.dim() == 2 && v.size(0) == 1) {
                // v: [1, flat] -> [1,1,*payload]
                std::vector<int64_t> payload_shape;
                for (int d = 2; d < A.dim(); ++d) payload_shape.push_back(A.size(d));
                v = v.view(payload_shape).unsqueeze(0).unsqueeze(0);
            }
        }

        if (A.dim() >= 2 && v.dim() == A.dim() - 2) {
            v = v.unsqueeze(0).unsqueeze(0); // -> [1,1,...]
        } else if (A.dim() >= 1 && v.dim() == A.dim() - 1) {
            v = v.unsqueeze(0); // -> [1,...]
        }

        // Ensure broadcasting works
        try {
            return v.expand_as(A);
        } catch (...) {
            return v;
        }
    };

    auto reduce_bias_like_A = [&](const torch::Tensor& term, const torch::Tensor& A) {
        // For A shaped [spec, batch, features] -> sum over dims [2..] -> [spec, batch]
        // For A shaped [spec, features] -> sum over dim 1 -> [spec], then expand to [spec, 1]
        if (!term.defined() || !A.defined()) return term;
        
        torch::Tensor result;
        if (A.dim() >= 3) {
            // 3D A: [spec, batch, features] or higher
            // Sum over feature dimensions [2..] to get [spec, batch]
            std::vector<int64_t> dims;
            for (int64_t d = 2; d < term.dim(); ++d) dims.push_back(d);
            result = dims.empty() ? term : term.sum(dims);
            // Ensure result is [spec, batch] - if it's already correct, keep it
            if (result.dim() == 2 && result.size(0) == A.size(0) && result.size(1) == A.size(1)) {
                return result;
            }
        } else if (A.dim() == 2) {
            // 2D A: [spec, features]
            // Sum over feature dimension to get [spec], then expand to [spec, 1]
            if (term.dim() >= 2) {
                result = term.sum({1}); // [spec]
            } else {
                result = term; // Already [spec]
            }
            // Expand to [spec, 1] to maintain [spec, batch] format (batch=1)
            if (result.dim() == 1) {
                result = result.unsqueeze(1); // [spec] -> [spec, 1]
            }
            return result;
        } else {
            // 1D A: [features] - should not happen in normal flow, but handle it
            result = term;
        }
        
        // Final check: ensure result is at least 2D [spec, batch]
        if (result.dim() == 1) {
            result = result.unsqueeze(1); // Add batch dimension
        }
        
        return result;
    };

    BoundA new_lA, new_uA;

    // ----- LOWER path -----
    if (effective_lA.defined()) {
        auto aL_l = relaxation_result.lb_lower_d.defined() ? relaxation_result.lb_lower_d : relaxation_result.d_lower;
        auto aU_l = relaxation_result.lb_upper_d.defined() ? relaxation_result.lb_upper_d : relaxation_result.d_upper;
        auto bL_l = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower : torch::zeros_like(input_lower);
        auto bU_l = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper : torch::zeros_like(input_lower);

        if (effective_lA.isTensor()) {
            torch::Tensor lA = effective_lA.asTensor();
            auto Apos = torch::clamp_min(lA, 0);
            auto Aneg = torch::clamp_max(lA, 0);

            auto A_l = Apos * expand_like(aL_l, lA) + Aneg * expand_like(aU_l, lA);
            auto b_l = Apos * expand_like(bL_l, lA) + Aneg * expand_like(bU_l, lA);

            new_lA = BoundA(A_l);
            auto add_lbias = reduce_bias_like_A(b_l, lA);
            lbias = lbias.defined() ? (lbias + add_lbias) : add_lbias;
        } else {
            // Patches mode
            auto patches = effective_lA.asPatches();
            
            // maybe_unfold_patches
            torch::Tensor aL_l_unfolded = maybe_unfold_patches(aL_l, effective_lA);
            torch::Tensor aU_l_unfolded = maybe_unfold_patches(aU_l, effective_lA);
            
            // Patches doesn't support clamp directly on object, need to use patches tensor
            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min(P, 0);
            torch::Tensor Pneg = torch::clamp_max(P, 0);
            
            // Multiply unfolded slopes
            // Shapes should match or broadcast
            torch::Tensor P_new = Ppos * aL_l_unfolded + Pneg * aU_l_unfolded;
            
            new_lA = BoundA(patches->create_similar(P_new));
            
            // Expand bias if [C, H, W] -> [1, C, H, W]
            if (bL_l.dim() == 3) bL_l = bL_l.unsqueeze(0);
            if (bU_l.dim() == 3) bU_l = bU_l.unsqueeze(0);
            
            // Unfold
            torch::Tensor bL_unfolded = inplace_unfold(bL_l, 
                {patches->patches.size(-2), patches->patches.size(-1)}, 
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            
            
            torch::Tensor bL_ready = bL_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
            // [1, batch, out_h, out_w, C, kh, kw]

            // Define bU_ready
            torch::Tensor bU_unfolded = inplace_unfold(bU_l,
                {patches->patches.size(-2), patches->patches.size(-1)},
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            torch::Tensor bU_ready = bU_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);

            // Ppos * bL_ready -> [out_c, batch, out_h, out_w, C, kh, kw]
            // Sum over C, kh, kw -> [out_c, batch, out_h, out_w]
            // Result is bias [batch, out_c, out_h, out_w]

            torch::Tensor term1 = (Ppos * bL_ready).sum({-3, -2, -1});
            torch::Tensor term2 = (Pneg * bU_ready).sum({-3, -2, -1});
            
            torch::Tensor total_bias = term1 + term2; // [out_c, batch, out_h, out_w]
            
            // Permute to [batch, out_c, out_h, out_w]
            total_bias = total_bias.permute({1, 0, 2, 3});
            
            lbias = lbias.defined() ? (lbias + total_bias) : total_bias;
        }
    }

    // ----- UPPER path -----
    if (effective_uA.defined()) {
        auto aU_u = relaxation_result.ub_upper_d.defined() ? relaxation_result.ub_upper_d : relaxation_result.d_upper;
        // Use ub_lower_d for upper-path A<0 (matches auto_LiRPA). Fall back to standard lower slope if not set.
        auto aL_u = relaxation_result.ub_lower_d.defined() ? relaxation_result.ub_lower_d
                                                          : relaxation_result.d_lower;
        auto bU_u = relaxation_result.bias_upper.defined() ? relaxation_result.bias_upper : torch::zeros_like(input_lower);
        auto bL_u = relaxation_result.bias_lower.defined() ? relaxation_result.bias_lower : torch::zeros_like(input_lower);

        if (effective_uA.isTensor()) {
            torch::Tensor uA = effective_uA.asTensor();
            auto Apos = torch::clamp_min(uA, 0);
            auto Aneg = torch::clamp_max(uA, 0);

            auto A_u = Apos * expand_like(aU_u, uA) + Aneg * expand_like(aL_u, uA);
            auto b_u = Apos * expand_like(bU_u, uA) + Aneg * expand_like(bL_u, uA);

            new_uA = BoundA(A_u);
            auto add_ubias = reduce_bias_like_A(b_u, uA);
            ubias = ubias.defined() ? (ubias + add_ubias) : add_ubias;
        } else {
            // Patches mode
            auto patches = effective_uA.asPatches();
            
            // maybe_unfold_patches
            torch::Tensor aU_u_unfolded = maybe_unfold_patches(aU_u, effective_uA);
            torch::Tensor aL_u_unfolded = maybe_unfold_patches(aL_u, effective_uA);
            
            torch::Tensor P = patches->patches;
            torch::Tensor Ppos = torch::clamp_min(P, 0);
            torch::Tensor Pneg = torch::clamp_max(P, 0);
            
            torch::Tensor P_new = Ppos * aU_u_unfolded + Pneg * aL_u_unfolded;
            new_uA = BoundA(patches->create_similar(P_new));
            
            // Bias
            if (bU_u.dim() == 3) bU_u = bU_u.unsqueeze(0);
            if (bL_u.dim() == 3) bL_u = bL_u.unsqueeze(0);
            
            torch::Tensor bU_unfolded = inplace_unfold(bU_u, 
                {patches->patches.size(-2), patches->patches.size(-1)}, 
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            torch::Tensor bU_ready = bU_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
            
            torch::Tensor bL_unfolded = inplace_unfold(bL_u, 
                {patches->patches.size(-2), patches->patches.size(-1)}, 
                patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
            torch::Tensor bL_ready = bL_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
            
            torch::Tensor term1 = (Ppos * bU_ready).sum({-3, -2, -1});
            torch::Tensor term2 = (Pneg * bL_ready).sum({-3, -2, -1});
            
            torch::Tensor total_bias = term1 + term2; 
            total_bias = total_bias.permute({1, 0, 2, 3});
            
            ubias = ubias.defined() ? (ubias + total_bias) : total_bias;
        }
    }

    // Ensure outputA_matrices has the right structure if not already set
    if (outputA_matrices.size() == 0) {
        outputA_matrices.append(Pair<BoundA, BoundA>(new_lA, new_uA));
    }
    
}

torch::Tensor BoundedReLUNode::maybe_unfold_patches(const torch::Tensor& d_tensor, const BoundA& last_A) {
    if (!d_tensor.defined() || !last_A.isPatches()) {
        return d_tensor;
    }
    auto patches = last_A.asPatches();
    
    // d_tensor shape: [N, C, H, W]
    // Needs to unfold to match patches kernel [C, kh, kw] at each location
    
    if (d_tensor.dim() == 3) {
        // [C, H, W] -> [1, C, H, W]
        return maybe_unfold_patches(d_tensor.unsqueeze(0), last_A);
    }
    
    // Use inplace_unfold
    torch::Tensor d_unfolded = inplace_unfold(d_tensor, 
        {patches->patches.size(-2), patches->patches.size(-1)}, 
        patches->stride, patches->padding, patches->inserted_zeros, patches->output_padding);
    
    // d_unfolded: [N, patches_h, patches_w, C, kh, kw]
    
    // Permute to match patches: [out_c, batch, out_h, out_w, C, kh, kw]
    // We need to broadcast/permute d_unfolded to this.
    // d_unfolded corresponds to batch, out_h, out_w...
    // P corresponds to out_c, batch, out_h, out_w...
    
    // Permute to [1, batch, patches_h, patches_w, C, kh, kw] (1 for out_c broadcast)
    // 0:N -> 1
    // 1:ph -> 2
    // 2:pw -> 3
    // 3:C -> 4
    // 4:kh -> 5
    // 5:kw -> 6
    
    return d_unfolded.permute({0, 1, 2, 3, 4, 5}).unsqueeze(0);
}

// IBP (Interval Bound Propagation): Fast interval-based bound computation for ReLU
BoundedTensor<torch::Tensor> BoundedReLUNode::computeIntervalBoundPropagation(
    const Vector<BoundedTensor<torch::Tensor>>& inputBounds) {
    
    if (inputBounds.size() < 1) {
        throw std::runtime_error("ReLU module requires at least one input");
    }
    
    const auto& inputBoundsPair = inputBounds[0];
    torch::Tensor inputLowerBound = inputBoundsPair.lower();
    torch::Tensor inputUpperBound = inputBoundsPair.upper();
    
    // Set input size from the input tensor during IBP
    if (_input_size == 0 && inputLowerBound.defined()) {
        _input_size = inputLowerBound.numel();
    }
    
    // ReLU: y = max(0, x)
    torch::Tensor lowerBound = torch::clamp_min(inputLowerBound, 0);  // max(0, lower)
    torch::Tensor upperBound = torch::clamp_min(inputUpperBound, 0);  // max(0, upper)
    
    // Set output size from the computed bounds during IBP
    if (_output_size == 0 && lowerBound.defined()) {
        _output_size = lowerBound.numel();
    }

    return BoundedTensor<torch::Tensor>(lowerBound, upperBound);
}

// Node information
unsigned BoundedReLUNode::getInputSize() const {
    return _input_size;
}

unsigned BoundedReLUNode::getOutputSize() const {
    // If output size is not set, try to infer from input size
    if (_output_size == 0 && _input_size > 0) {
        return _input_size; // ReLU preserves input size
    }
    // If still 0, return a reasonable default for testing
    if (_output_size == 0) {
        return 2; // Default size for testing
    }
    return _output_size;
}

void BoundedReLUNode::setInputSize(unsigned size) {
    _input_size = size;
}

void BoundedReLUNode::setOutputSize(unsigned size) {
    _output_size = size;
}

// Unified backward relaxation method (following auto_LiRPA approach)
BoundedReLUNode::RelaxationResult BoundedReLUNode::_backwardRelaxation(
    const BoundA& last_lA, const BoundA& last_uA,
    const torch::Tensor& input_lower, const torch::Tensor& input_upper)
{
    RelaxationResult result;

    // Compute standard CROWN upper bound relaxation (secant line for unstable neurons)
    auto [upper_d, upper_b] = _reluUpperBound(input_lower, input_upper);

    // Compute standard CROWN lower bound
    torch::Tensor lower_d = _computeStandardCROWNLowerBound(input_lower, input_upper);

    // Store slopes for alpha initialization if needed
    init_d = lower_d.detach().clone();
    init_upper_d = upper_d.detach().clone();

    // Set basic relaxation slopes and biases
    result.d_lower = lower_d;
    result.d_upper = upper_d;
    // For lower bound: bias_lower must be zero (all ReLU lower relaxations pass through origin: y ≥ α·x + 0)
    result.bias_lower = torch::zeros_like(input_lower);
    result.bias_upper = upper_b;

    // Apply alpha masking if enabled
    _maskAlpha(input_lower, input_upper, upper_d, result);

    return result;
}

// Helper method: CROWN upper bound computation
std::pair<torch::Tensor, torch::Tensor> BoundedReLUNode::_reluUpperBound(const torch::Tensor& lb, const torch::Tensor& ub)
{
    // Compute standard CROWN upper bound relaxation slopes
    torch::Tensor lb_r = torch::clamp_max(lb, 0);  // Negative part of lower bound
    torch::Tensor ub_r = torch::clamp_min(ub, 0);  // Positive part of upper bound
    ub_r = torch::max(ub_r, lb_r + 1e-8);  // Avoid division by zero

    // Standard CROWN upper bound slope formula: upper_bound / (upper_bound - lower_bound)
    torch::Tensor upper_d = ub_r / (ub_r - lb_r);
    torch::Tensor upper_b = -lb_r * upper_d;

    return std::make_pair(upper_d, upper_b);
}

// Helper method: Standard CROWN lower bound computation
torch::Tensor BoundedReLUNode::_computeStandardCROWNLowerBound(const torch::Tensor& input_lower, const torch::Tensor& input_upper)
{
    // Initialize slopes for the three ReLU cases
    torch::Tensor slopes_lower = torch::zeros_like(input_lower);

    // Case 1: input_lower >= 0 (always active) - slope = 1
    auto always_active_mask = input_lower >= 0;
    slopes_lower = torch::where(always_active_mask, torch::ones_like(slopes_lower), slopes_lower);

    // Case 2: input_upper <= 0 (always inactive) - slope = 0 (already initialized)
    // Case 3: uncertain neurons - use adaptive approach
    auto uncertain_mask = (input_lower < 0) & (input_upper > 0);
    if ( uncertain_mask.any().item<bool>() )
    {
        // TODO: Avoid recomputing this and instead pass through by reference
        // Compute upper slope for adaptive decision
        torch::Tensor ub_r = torch::clamp_min(input_upper, 0);
        torch::Tensor lb_r = torch::clamp_max(input_lower, 0);
        ub_r = torch::max(ub_r, lb_r + 1e-8);
        torch::Tensor upper_slope = ub_r / (ub_r - lb_r);

        // if upper_slope > 0.5, use slope = 1, otherwise use slope = 0
        auto adaptive_mask = upper_slope > 0.5;
        torch::Tensor lower_slope = torch::where(adaptive_mask,
                                                torch::ones_like(input_lower),
                                                torch::zeros_like(input_lower));
        slopes_lower = torch::where(uncertain_mask, lower_slope, slopes_lower);
    }

    return slopes_lower;
}

// Helper method: Get alpha parameters for specific bound type
// NOTE: This method is deprecated - alpha is now fetched directly in boundBackward at multiply time
// Apply masking for stable/unstable neurons (following auto_LiRPA approach)
// FIXED: Properly handle unstable-only alpha tensors and shape broadcasting
void BoundedReLUNode::_maskAlpha(const torch::Tensor& input_lower, const torch::Tensor& input_upper, const torch::Tensor& upper_d, RelaxationResult& result)
{
    // Compute neuron status masks (flattened to match neuron dimension)
    auto input_lb_flat = input_lower.flatten();
    auto input_ub_flat = input_upper.flatten();
    auto always_active_mask = input_lb_flat >= 0;      // [outDim]
    auto always_inactive_mask = input_ub_flat <= 0;    // [outDim]
    auto unstable = (input_lb_flat < 0) & (input_ub_flat > 0); // [outDim]
    int outDim = (int)input_lb_flat.numel();

    // For alpha optimization - alpha is the optimizable slope
    // Apply alpha optimization for ALL backward passes (output AND intermediate)
    // Each backward pass uses alpha keyed by startKey, enabling per-target optimization
    if (isAlphaOptimizationEnabled() && _alphaCrownAnalysis
        && (_optimizationStage == "opt" || _optimizationStage == "reuse")
        && _currentSpecDim > 0) {
        auto* crown = _alphaCrownAnalysis->getCROWNAnalysis();
        std::string startKey = crown->currentStartKey();
        if (startKey.empty()) startKey = "default";

        // DEBUG: Print alpha application for every ReLU during backward pass
        if (LunaConfiguration::VERBOSE) {
            printf("[DEBUG _maskAlpha] node=%u, startKey=%s: APPLYING optimized alpha\n",
                   getNodeIndex(), startKey.c_str());
        }

        int specDim = _currentSpecDim;
        Vector<unsigned> currentSpecIndices;
        bool hasSpecLookup = false;
        Vector<unsigned> cachedSpecIndices;
        bool cachedSparseMode = false;
        unsigned cachedNodeSize = 0;
        if (crown) {
            currentSpecIndices = crown->currentStartSpecIndices();
            hasSpecLookup = crown->getAlphaStartCacheInfo(startKey, cachedSpecIndices, cachedSparseMode, cachedNodeSize);
        }

        // Get alpha result (now returns AlphaResult with unstable-only alpha)
        // Alpha shape: [2, spec, numUnstable] — dim 0 separates lA/uA paths
        auto alphaResult = _alphaCrownAnalysis->getAlphaForNodeAllSpecs(
            getNodeIndex(),
            startKey, specDim, outDim,
            input_lower, input_upper);

        if (LunaConfiguration::VERBOSE && alphaResult.alpha.defined()) {
            printf("[DEBUG _maskAlpha] node=%u, alpha shape=[%lld,%lld,%lld], numUnstable=%d, outDim=%d\n",
                   getNodeIndex(),
                   alphaResult.alpha.dim() >= 1 ? (long long)alphaResult.alpha.size(0) : 0,
                   alphaResult.alpha.dim() >= 2 ? (long long)alphaResult.alpha.size(1) : 0,
                   alphaResult.alpha.dim() >= 3 ? (long long)alphaResult.alpha.size(2) : 0,
                   alphaResult.numUnstable, alphaResult.outDim);
        }

        if (alphaResult.numUnstable > 0 && alphaResult.alpha.defined() && alphaResult.alpha.numel() > 0) {
            // Clone alpha to create a fresh tensor for this iteration's computation graph.
            // This is necessary because alphaResult.alpha is a view of the parameter tensor
            // that persists across iterations. Using it directly would cause either:
            // 1. "modified by inplace operation" errors (if we clamp it)
            // 2. "backward through graph a second time" errors (if we reuse graph nodes)
            // The clone() operation preserves gradient flow - gradients from the clone
            // will flow back to the original alpha parameter during backward().
            auto alpha_unstable = alphaResult.alpha.clone(); // [2, spec or spec+1, numUnstable]

            // If alpha has a default spec slot, map current spec indices into compact alpha spec indices.
            if (alphaResult.hasSpecDefaultSlot && hasSpecLookup && cachedSparseMode && cachedNodeSize > 0) {
                // Build lookup: size [nodeSize], default 0
                auto lookup = torch::zeros({(long long)cachedNodeSize},
                                           torch::TensorOptions().dtype(torch::kLong).device(alpha_unstable.device()));
                for (int i = 0; i < (int)cachedSpecIndices.size(); ++i) {
                    unsigned idx = cachedSpecIndices[i];
                    if (idx < cachedNodeSize) {
                        lookup[idx] = i + 1; // 1..k, 0 is default slot
                    }
                }

                // Map current spec indices (if available) into compact indices
                if (currentSpecIndices.size() > 0) {
                    auto idxTensor = torch::empty({(long long)currentSpecIndices.size()},
                                                  torch::TensorOptions().dtype(torch::kLong).device(alpha_unstable.device()));
                    for (int i = 0; i < (int)currentSpecIndices.size(); ++i) {
                        unsigned idx = currentSpecIndices[i];
                        idxTensor[i] = (idx < cachedNodeSize)
                            ? static_cast<int64_t>(lookup[idx].item<int64_t>())
                            : static_cast<int64_t>(0);
                    }
                    // dim 1 is the spec dimension (alpha shape: [2, spec, numUnstable])
                    alpha_unstable = alpha_unstable.index_select(1, idxTensor);
                    specDim = (int)alpha_unstable.size(1);
                }
            }

            // Create full alpha tensor [spec, outDim] using functional operations only.
            // IMPORTANT: Avoid in-place operations (index_put_, scatter_) which create
            // IndexPutBackward0 nodes that can cause "backward through graph a second time"
            // errors when tensors are reused across optimization iterations.
            auto options = alpha_unstable.options();

            // dim 1 is the spec dimension (alpha shape: [2, spec, numUnstable])
            int alphaSpecDim = (int)(alpha_unstable.defined() ? alpha_unstable.size(1) : specDim);
            specDim = alphaSpecDim;

            // Expand masks to [spec, outDim] for broadcasting
            auto always_active_expanded = always_active_mask.unsqueeze(0).expand({specDim, outDim});

            // Split alpha into lA-path (alpha[0]) and uA-path (alpha[1])
            // matching auto-LiRPA: lb_lower_d = selected_alpha[0], ub_lower_d = selected_alpha[1]
            auto alpha_lower_unstable = alpha_unstable[0]; // [specDim, numUnstable]
            auto alpha_upper_unstable = alpha_unstable[1]; // [specDim, numUnstable]

            // Build alpha_full using scatter (non-in-place version returns new tensor)
            // Expand indices from [numUnstable] to [specDim, numUnstable] for scatter
            auto indices = alphaResult.unstableIndices.unsqueeze(0).expand({specDim, alphaResult.numUnstable});

            // scatter() returns a new tensor (not in-place like scatter_())
            // Build separate full tensors for lA and uA paths
            torch::Tensor alpha_lower_full = torch::zeros({specDim, outDim}, options).scatter(1, indices, alpha_lower_unstable);
            torch::Tensor alpha_upper_full = torch::zeros({specDim, outDim}, options).scatter(1, indices, alpha_upper_unstable);

            // Apply always_active mask: set those neurons to 1.0
            alpha_lower_full = torch::where(always_active_expanded,
                                      torch::ones({specDim, outDim}, options),
                                      alpha_lower_full);
            alpha_upper_full = torch::where(always_active_expanded,
                                      torch::ones({specDim, outDim}, options),
                                      alpha_upper_full);

            // Per-spec "upper" slope for the A<0 branch (secant slope)
            // Apply stable neuron masking (same as standard CROWN path)
            auto upper_d_flat = upper_d.flatten();
            auto upper_d_masked = torch::where(always_active_mask, torch::ones_like(upper_d_flat), upper_d_flat);
            upper_d_masked = torch::where(always_inactive_mask, torch::zeros_like(upper_d_masked), upper_d_masked);
            auto k_upper_spec = upper_d_masked.unsqueeze(0).expand({specDim, outDim}); // [spec, outDim]

            // Write per-spec lower-path choices (alpha[0] for lA path)
            result.lb_lower_d = alpha_lower_full;  // used when A >= 0
            result.lb_upper_d = k_upper_spec;      // used when A < 0

            // Write per-spec upper-path choices (alpha[1] for uA path)
            result.ub_upper_d = k_upper_spec;      // used when A >= 0
            // Match auto_LiRPA: ub_lower_d = selected_alpha[1] — independent slope for uA path
            result.ub_lower_d = alpha_upper_full;  // used when A < 0


            // Biases (shape: [outDim])
            auto b_upper = -input_lb_flat * upper_d_flat; // secant bias
            // Apply stable neuron masking to bias_upper (same as standard CROWN path)
            auto b_upper_masked = torch::where(always_active_mask | always_inactive_mask,
                                               torch::zeros_like(b_upper),
                                               b_upper);
            result.bias_lower = torch::zeros_like(input_lb_flat); // for Apos (alpha) branch
            result.bias_upper = b_upper_masked;                    // for Aneg (secant) branch



            // Skip the standard masking below since we've already set everything
            return;
        }
    }
}

// Alpha-aware relaxation computation (implementing BoundedAlphaOptimizeNode interface)
void BoundedReLUNode::computeAlphaRelaxation(
    const torch::Tensor& last_lA,
    const torch::Tensor& last_uA,
    const torch::Tensor& input_lower,
    const torch::Tensor& input_upper,
    torch::Tensor& d_lower,
    torch::Tensor& d_upper,
    torch::Tensor& bias_lower,
    torch::Tensor& bias_upper) {
    
    // Use the unified backward relaxation method
    // Wrap tensors in BoundA
    auto result = _backwardRelaxation(BoundA(last_lA), BoundA(last_uA), input_lower, input_upper);
    
    // Extract the computed slopes and biases
    d_lower = result.d_lower;
    d_upper = result.d_upper;
    bias_lower = result.bias_lower;
    bias_upper = result.bias_upper;
}

// Get CROWN slopes for alpha initialization (following auto_LiRPA approach)
torch::Tensor BoundedReLUNode::getCROWNSlope(bool isLowerBound) const
{
    if (isLowerBound) {
        if (!hasInitD()) {
            // Return default lower slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 0.5f, torch::kFloat32);
        }
        return init_d;  // Return actual lower bound slopes
    } else {
        if (!init_upper_d.defined() || init_upper_d.numel() == 0) {
            // Return default upper slopes if CROWN slopes not available
            return torch::full({getOutputSize()}, 1.0f, torch::kFloat32);
        }
        return init_upper_d;  // Return actual upper bound slopes
    }
}

} // namespace NLR
