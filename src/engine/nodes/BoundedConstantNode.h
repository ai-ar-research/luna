/*********************                                                        */
/*! \file BoundedConstantNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_CONSTANT_NODE_H__
#define __BOUNDED_CONSTANT_NODE_H__

#include "BoundedTorchNode.h"

namespace NLR {

class BoundedConstantNode : public NLR::BoundedTorchNode {
public:
    BoundedConstantNode(const torch::Tensor& constantValue, const String& name = "");

    NLR::NodeType getNodeType() const override { return NLR::NodeType::CONSTANT; }
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

    unsigned getInputSize() const override { return 0; }
    unsigned getOutputSize() const override { return _constantValue.numel(); }
    bool isPerturbed() const override { return false; }

    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;

    torch::Tensor getConstantValue() const { return _constantValue; }

private:
    torch::Tensor _constantValue;
};

} // namespace NLR

#endif // __BOUNDED_CONSTANT_NODE_H__