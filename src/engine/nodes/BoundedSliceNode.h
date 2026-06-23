/*********************                                                        */
/*! \file BoundedSliceNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_SLICE_NODE_H__
#define __BOUNDED_SLICE_NODE_H__

#include "BoundedTorchNode.h"
#include <vector>

namespace NLR {

class BoundedSliceNode : public BoundedTorchNode {
public:
    BoundedSliceNode(int64_t start, int64_t end, int64_t axes, int64_t steps = 1,
                     const String& name = "");

    NodeType getNodeType() const override { return NodeType::SLICE; }
    String getNodeName() const override { return _nodeName; }
    unsigned getNodeIndex() const override { return _nodeIndex; }

    torch::Tensor forward(const torch::Tensor& input) override;
    torch::Tensor forward(const std::vector<torch::Tensor>& inputs) override;

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

    int64_t getStart() const { return _start; }
    int64_t getEnd() const { return _end; }
    int64_t getAxes() const { return _axes; }
    int64_t getSteps() const { return _steps; }

    void setInputShape(const std::vector<int64_t>& shape) { _input_shape = shape; }
    const std::vector<int64_t>& getInputShape() const { return _input_shape; }

private:
    std::pair<int64_t, int64_t> _fixup_params(const std::vector<int64_t>& shape,
                                               int64_t start, int64_t end,
                                               int64_t axes, int64_t steps) const;

    int64_t _start;
    int64_t _end;
    int64_t _axes;
    int64_t _steps;

    std::vector<int64_t> _input_shape;

    String _nodeName;
    unsigned _nodeIndex;
    unsigned _input_size;
    unsigned _output_size;
};

} // namespace NLR

#endif // __BOUNDED_SLICE_NODE_H__
