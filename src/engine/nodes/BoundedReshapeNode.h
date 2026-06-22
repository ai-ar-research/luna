/*********************                                                        */
/*! \file BoundedReshapeNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_RESHAPE_NODE_H__
#define __BOUNDED_RESHAPE_NODE_H__

#include "BoundedTorchNode.h"
#include "../input_parsers/Operations.h"

namespace NLR {

class BoundedReshapeNode : public BoundedTorchNode {
public:
    BoundedReshapeNode(const Operations::ReshapeWrapper& reshape_module);

    NLR::NodeType getNodeType() const override { return NLR::NodeType::RESHAPE; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    torch::Tensor forward(const torch::Tensor& input) override;

    Pair<torch::Tensor, torch::Tensor> computeCrownBackwardPropagation(const torch::Tensor& lastLowerAlpha,
                                               const torch::Tensor& lastUpperAlpha,
                                               const Vector<BoundedTensor<torch::Tensor>>& inputs);

    BoundedTensor<torch::Tensor> computeIntervalBoundPropagation(
        const Vector<BoundedTensor<torch::Tensor>>& inputs) override;

    void boundBackward(
        const BoundA& last_lA,
        const BoundA& last_uA,
        const Vector<BoundedTensor<torch::Tensor>>& inputBounds,
        Vector<Pair<BoundA, BoundA>>& outputA_matrices,
        torch::Tensor& lbias,
        torch::Tensor& ubias
    ) override;

    unsigned getInputSize() const override { return _input_size; }
    unsigned getOutputSize() const override { return _output_size; }
    bool isPerturbed() const override { return true; }
    String getModuleType() const { return "Reshape"; }

    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }

    void setInputShape(const std::vector<int64_t>& shape) { _input_shape = shape; }
    const std::vector<int64_t>& getInputShape() const { return _input_shape; }

private:
    Operations::ReshapeWrapper _reshape_module;
    std::vector<int64_t> _input_shape;
};

} // namespace NLR

#endif // __BOUNDED_RESHAPE_NODE_H__
