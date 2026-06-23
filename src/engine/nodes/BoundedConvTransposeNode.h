/*********************                                                        */
/*! \file BoundedConvTransposeNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_CONV_TRANSPOSE_NODE_H__
#define __BOUNDED_CONV_TRANSPOSE_NODE_H__

#include "BoundedTorchNode.h"
#include "conv/ConvolutionMode.h"
#include <vector>

namespace NLR {

class BoundedConvTransposeNode : public BoundedTorchNode {
public:
    BoundedConvTransposeNode(const torch::nn::ConvTranspose2d& convTransposeModule,
                             ConvMode mode = ConvMode::MATRIX,
                             const String& name = "");

    NodeType getNodeType() const override { return NodeType::CONVTRANSPOSE; }
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

    unsigned inferOutputSize(unsigned inputSize) const;

    void setInputSize(unsigned size) override;
    void setOutputSize(unsigned size) override;

    void setNodeIndex(unsigned index) override { _nodeIndex = index; }
    void setNodeName(const String& name) override { _nodeName = name; }
    void moveToDevice(const torch::Device& device) override;

    std::vector<int> getPadding() const { return padding; }
    std::vector<int> getStride() const { return stride; }
    std::vector<int> getDilation() const { return dilation; }
    std::vector<int> getOutputPadding() const { return output_padding; }
    int getGroups() const { return groups; }
    bool hasBias() const { return has_bias; }
    ConvMode getMode() const { return mode; }

private:
    void initializeFromConvTranspose2d(const torch::nn::ConvTranspose2d& convTransposeModule);

    BoundA boundOneSide(const BoundA& last_A,
                        const torch::Tensor& weight,
                        const torch::Tensor& bias,
                        torch::Tensor& sum_bias);

    torch::nn::ConvTranspose2d convtranspose2d{nullptr};

    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    std::vector<int> output_padding;
    int groups;
    bool has_bias;

    ConvMode mode;

    std::vector<int> input_shape;
    std::vector<int> output_shape;
};

} // namespace NLR

#endif // __BOUNDED_CONV_TRANSPOSE_NODE_H__
