/*********************                                                        */
/*! \file MatrixConvolution.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __MATRIX_CONVOLUTION_H__
#define __MATRIX_CONVOLUTION_H__

#include <torch/torch.h>
#include <vector>

namespace NLR {

class MatrixConvolution {
public:
    static torch::Tensor im2col(
        const torch::Tensor& input,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        int groups = 1
    );

    static torch::Tensor col2im(
        const torch::Tensor& col,
        const std::vector<int>& output_size,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        int groups = 1
    );

    static std::vector<int> computeConvOutputShape(
        const std::vector<int>& input_shape,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    static std::vector<int> computeTransposeOutputPadding(
        const std::vector<int>& input_shape,
        const std::vector<int>& output_shape,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    static torch::Tensor matrixConvForward(
        const torch::Tensor& input_matrix,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        const std::vector<int>& output_shape
    );

    static torch::Tensor matrixConvBackward(
        const torch::Tensor& grad_output,
        const torch::Tensor& weight,
        const std::vector<int>& input_shape,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation,
        const std::vector<int>& output_padding
    );

private:
    static torch::Tensor unfoldInput(
        const torch::Tensor& input,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding,
        const std::vector<int>& dilation
    );

    static torch::Tensor foldColumns(
        const torch::Tensor& col,
        const std::vector<int>& output_size,
        const std::vector<int>& kernel_size,
        const std::vector<int>& stride,
        const std::vector<int>& padding
    );
};

} // namespace NLR

#endif // __MATRIX_CONVOLUTION_H__
