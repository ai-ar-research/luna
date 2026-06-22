/*********************                                                        */
/*! \file BoundedLinearNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_LINEAR_NODE_H__
#define __BOUNDED_LINEAR_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedLinearNode : public NLR::BoundedTorchNode {
public:
    BoundedLinearNode(const torch::nn::Linear& linearModule,
        float alpha = 1.0f, const String& name = "");

    NLR::NodeType getNodeType() const override { return NLR::NodeType::LINEAR; }
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

    bool hasInputSize() const { return _input_size > 0; }
    bool hasOutputSize() const { return _output_size > 0; }

    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;

    torch::Tensor computeLinearIBPLowerBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound);
    torch::Tensor computeLinearIBPUpperBound(const torch::Tensor& inputLowerBound, const torch::Tensor& inputUpperBound);

    const torch::nn::Linear& getLinearModule() const { return _linearModule; }

private:
    torch::nn::Linear _linearModule;
    float _alpha;

    // Avoids repeated .to(device) calls
    mutable torch::Tensor _cached_weight;    // alpha * weight, on target device
    mutable torch::Tensor _cached_bias;
    mutable torch::Device _cached_device{torch::kCPU};

    // Fast no-op if already cached
    void ensureWeightsOnDevice(const torch::Device& device) const;
};

} // namespace NLR

#endif // __BOUNDED_LINEAR_NODE_H__
