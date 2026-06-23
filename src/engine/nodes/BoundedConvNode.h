/*********************                                                        */
/*! \file BoundedConvNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BOUNDED_CONV_NODE_H__
#define __BOUNDED_CONV_NODE_H__

#include "BoundedTorchNode.h"
#include "conv/ConvolutionMode.h"
#include <vector>

namespace NLR {

class BoundedConvNode : public BoundedTorchNode {
public:
    BoundedConvNode(const torch::nn::Conv1d& convModule,
                     ConvMode mode = ConvMode::MATRIX,
                     const String& name = "");

    BoundedConvNode(const torch::nn::Conv2d& convModule,
                     ConvMode mode = ConvMode::MATRIX,
                     const String& name = "");

    NodeType getNodeType() const override { return NodeType::CONV; }
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
    int getGroups() const { return groups; }
    bool hasBias() const { return has_bias; }
    ConvMode getMode() const { return mode; }

    void setReluFollowed(bool followed) { relu_followed = followed; }
    bool isReluFollowed() const { return relu_followed; }

    void setInputShape(const std::vector<int>& shape) { input_shape = shape; }
    void setOutputShape(const std::vector<int>& shape) { output_shape = shape; }
    const std::vector<int>& getInputShape() const { return input_shape; }
    const std::vector<int>& getOutputShape() const { return output_shape; }

private:
    void initializeFromConv1d(const torch::nn::Conv1d& convModule);
    void initializeFromConv2d(const torch::nn::Conv2d& convModule);

    BoundA boundOneSide(const BoundA& last_A,
                        const torch::Tensor& weight,
                        const torch::Tensor& bias,
                        torch::Tensor& sum_bias);

    std::vector<int> computeOutputPadding(const std::vector<int>& input_shape,
                                           const std::vector<int>& output_shape,
                                           const torch::Tensor& weight) const;

    int conv_dim;

    torch::nn::Conv1d conv1d{nullptr};
    torch::nn::Conv2d conv2d{nullptr};

    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    int groups;
    bool has_bias;

    // Avoids int->int64_t conversion overhead in hot path
    std::vector<int64_t> stride_64;
    std::vector<int64_t> padding_64;
    std::vector<int64_t> dilation_64;

    ConvMode mode;
    bool relu_followed;
    bool patches_start;

    std::vector<int> input_shape;
    std::vector<int> output_shape;

    // Avoids repeated .to() conversions
    mutable torch::Tensor _cached_weight;
    mutable torch::Tensor _cached_bias;
    mutable torch::Device _cached_device{torch::kCPU};

    void ensureWeightsOnDevice(const torch::Device& device) const;
};

} // namespace NLR

#endif // __BOUNDED_CONV_NODE_H__
