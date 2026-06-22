/*********************                                                        */
/*! \file BoundedSigmoidNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_SIGMOID_NODE_H__
#define __BOUNDED_SIGMOID_NODE_H__

#include "BoundedAlphaOptimizedNode.h"
#include "configuration/LunaConfiguration.h"

namespace NLR {

class BoundedSigmoidNode : public NLR::BoundedAlphaOptimizeNode {
public:
    BoundedSigmoidNode(const torch::nn::Sigmoid& sigmoidModule, const String& name = "");

    NLR::NodeType getNodeType() const override { return NLR::NodeType::SIGMOID; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    torch::Tensor forward(const torch::Tensor& input) override;

    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds) override;

    unsigned getInputSize() const override;
    unsigned getOutputSize() const override;
    bool isPerturbed() const override { return true; }

    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;

    struct RelaxationResult {
        torch::Tensor d_lower;
        torch::Tensor d_upper;
        torch::Tensor bias_lower;
        torch::Tensor bias_upper;

        torch::Tensor lb_lower_d, ub_lower_d;
        torch::Tensor lb_upper_d, ub_upper_d;
        torch::Tensor lb_lower_b, ub_lower_b;
        torch::Tensor lb_upper_b, ub_upper_b;
    };

    RelaxationResult _backwardRelaxation(const BoundA& last_lA, const BoundA& last_uA,
                                        const torch::Tensor& input_lower, const torch::Tensor& input_upper);

    void computeAlphaRelaxation(
        const torch::Tensor& last_lA,
        const torch::Tensor& last_uA,
        const torch::Tensor& input_lower,
        const torch::Tensor& input_upper,
        torch::Tensor& d_lower,
        torch::Tensor& d_upper,
        torch::Tensor& bias_lower,
        torch::Tensor& bias_upper) override;

    torch::Tensor getCROWNSlope(bool isLowerBound) const override;

    std::pair<torch::Tensor, torch::Tensor> getDefaultTangentPoints(
        const torch::Tensor& lower, const torch::Tensor& upper);

private:
    std::shared_ptr<torch::nn::Sigmoid> _sigmoidModule;

    // Precomputed lookup tables (matching Python BoundTanh)
    torch::Tensor d_lower;
    torch::Tensor d_upper;
    torch::Tensor dfunc_values;

    double step_pre = 0.01;
    int num_points_pre = 0;
    double x_limit = 20.0;
    bool _lookupTablesInitialized = false;

    void precomputeRelaxation(double required_limit = -1.0);
    void precomputeDfuncValues();
    torch::Tensor retrieveFromPrecompute(const torch::Tensor& precomputed_d,
                                         const torch::Tensor& input_bound,
                                         const torch::Tensor& default_d);
    std::pair<torch::Tensor, torch::Tensor> generateDLowerUpper(const torch::Tensor& lower,
                                                                 const torch::Tensor& upper);
    std::pair<torch::Tensor, torch::Tensor> retrieveDFromK(const torch::Tensor& k);

    torch::Tensor sigmoidFunc(const torch::Tensor& x);
    torch::Tensor dsigmoidFunc(const torch::Tensor& x);

    void boundRelaxImpl(const torch::Tensor& input_lower, const torch::Tensor& input_upper);

    torch::Tensor maybe_unfold_patches(const torch::Tensor& d_tensor, const BoundA& last_A);

    // CROWN slopes for alpha initialization
    torch::Tensor init_lower_d;
    torch::Tensor init_upper_d;

    int _currentSpecDim{1};

    torch::Tensor mask_pos;
    torch::Tensor mask_neg;
    torch::Tensor mask_both;

    torch::Tensor lw, lb;
    torch::Tensor uw, ub;
};

} // namespace NLR

#endif // __BOUNDED_SIGMOID_NODE_H__
