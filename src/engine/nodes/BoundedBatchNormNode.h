/*********************                                                        */
/*! \file BoundedBatchNormNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_BATCHNORM_NODE_H__
#define __BOUNDED_BATCHNORM_NODE_H__

#include "BoundedTorchNode.h"
#include "../conv/Patches.h"

namespace NLR {

// y = scale * (x - mean) / sqrt(var + eps) + B
class BoundedBatchNormNode : public BoundedTorchNode {
public:
    BoundedBatchNormNode(
        const torch::Tensor& scale,
        const torch::Tensor& B,
        const torch::Tensor& mean,
        const torch::Tensor& var,
        float eps,
        const String& name = ""
    );

    NLR::NodeType getNodeType() const override { return NLR::NodeType::BATCHNORM; }
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

    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }

    void setInputSize(unsigned size) override { _input_size = size; }
    void setOutputSize(unsigned size) override { _output_size = size; }

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;

    torch::Tensor getScale() const { return _scale; }
    torch::Tensor getBias() const { return _bias; }
    torch::Tensor getMean() const { return _mean; }
    torch::Tensor getVar() const { return _var; }
    float getEps() const { return _eps; }

    const std::vector<int64_t>& getLastInputShape() const { return _last_input_shape; }

    torch::Tensor getTmpWeight(const torch::Tensor& like) const { return tmp_weight(like); }
    torch::Tensor getTmpBias(const torch::Tensor& like) const { return tmp_bias(like); }

private:
    torch::Tensor _scale;
    torch::Tensor _bias;
    torch::Tensor _mean;
    torch::Tensor _var;
    float _eps;

    std::vector<int64_t> _last_input_shape;

    // Avoids repeated .to(device) + sqrt per call
    mutable torch::Tensor _cached_tmp_weight;
    mutable torch::Tensor _cached_tmp_bias;
    mutable torch::Device _cached_device{torch::kCPU};

    void ensureCachedParams(const torch::Device& device) const;

    torch::Tensor tmp_weight(const torch::Tensor& like) const;
    torch::Tensor tmp_bias(const torch::Tensor& like) const;

    torch::Tensor broadcast_channel_param(const torch::Tensor& param, const torch::Tensor& x) const;

    BoundA boundOneSideTensor(
        const BoundA& last_A,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        torch::Tensor& sum_bias
    );

    BoundA boundOneSidePatches(
        const BoundA& last_A,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        torch::Tensor& sum_bias
    );
};

} // namespace NLR

#endif // __BOUNDED_BATCHNORM_NODE_H__
