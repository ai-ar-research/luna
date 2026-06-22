/*********************                                                        */
/*! \file Operations.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include <torch/torch.h>

namespace Operations {

class ReshapeImpl : public torch::nn::Module {
public:
    ReshapeImpl() {}
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& shape_tensor);
};
TORCH_MODULE(Reshape);

class ReshapeWrapper : public torch::nn::Module {
private:
    torch::Tensor shape_tensor;
public:
    ReshapeWrapper(torch::Tensor shape) : shape_tensor(shape) {
        register_buffer("shape", this->shape_tensor);
    }
    torch::Tensor forward(const torch::Tensor& input) {
        torch::Tensor flattened_shape = shape_tensor.flatten();
        std::vector<int64_t> new_shape;
        for (int64_t i = 0; i < flattened_shape.numel(); ++i) {
            new_shape.push_back(flattened_shape[i].item<int64_t>());
        }

        // preserve batch dimension
        if (input.dim() > 0 && input.size(0) == 1) {
            new_shape.insert(new_shape.begin(), 1);
        }

        return input.reshape(new_shape);
    }
};

class FlattenWrapper : public torch::nn::Module {
private:
    int64_t axis;
public:
    FlattenWrapper(int64_t axis_val = 1) : axis(axis_val) {}

    torch::Tensor forward(const torch::Tensor& input) {
        int64_t actual_axis = axis;
        if (actual_axis < 0) {
            actual_axis = input.dim() + actual_axis;
        }

        actual_axis = std::max(int64_t(0), std::min(actual_axis, int64_t(input.dim())));

        int64_t dim1 = 1;
        for (int64_t i = 0; i < actual_axis; ++i) {
            dim1 *= input.size(i);
        }

        int64_t dim2 = 1;
        for (int64_t i = actual_axis; i < input.dim(); ++i) {
            dim2 *= input.size(i);
        }

        if (dim1 == 1) {
            return input.reshape({dim2});
        } else if (dim2 == 1) {
            return input.reshape({dim1});
        }

        return input.reshape({dim1, dim2});
    }
};

class Constant : public torch::nn::Module {
    torch::Tensor value;
public:
    Constant(torch::Tensor value) : value(value) {
        register_buffer("value", this->value);
    }
    torch::Tensor forward();
};

} // namespace Operations

#endif // __OPERATIONS_H__ 