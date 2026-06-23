#ifndef __TENSOR_COMPARATORS_H__
#define __TENSOR_COMPARATORS_H__

#include "src/common/BoundedTensor.h"
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <string>
#include <sstream>

namespace test {

/**
 * Utilities for comparing tensors and bounds with tolerances.
 */
class TensorComparator {
public:
    // Compare two tensors element-wise with absolute and relative tolerance
    static bool allClose(const torch::Tensor& t1, const torch::Tensor& t2,
                        double atol = 1e-5, double rtol = 1e-4);

    // Get maximum absolute difference between two tensors
    static double maxAbsDiff(const torch::Tensor& t1, const torch::Tensor& t2);

    // Get detailed diff report as string
    static std::string diffReport(const torch::Tensor& t1, const torch::Tensor& t2,
                                  double atol = 1e-5, double rtol = 1e-4);

    // Compare bounded tensors
    static bool boundsClose(const BoundedTensor<torch::Tensor>& b1,
                           const BoundedTensor<torch::Tensor>& b2,
                           double atol = 1e-5, double rtol = 1e-4);

    // Get detailed diff report for bounds
    static std::string boundsDiffReport(const BoundedTensor<torch::Tensor>& b1,
                                       const BoundedTensor<torch::Tensor>& b2,
                                       double atol = 1e-5, double rtol = 1e-4);
};

} // namespace test

// Custom Google Test macros for tensor comparison
#define EXPECT_TENSORS_CLOSE(t1, t2, atol, rtol) \
    EXPECT_TRUE(test::TensorComparator::allClose(t1, t2, atol, rtol))

#define EXPECT_BOUNDS_CLOSE(b1, b2, atol, rtol) \
    EXPECT_TRUE(test::TensorComparator::boundsClose(b1, b2, atol, rtol))

#define ASSERT_TENSORS_CLOSE(t1, t2, atol, rtol) \
    ASSERT_TRUE(test::TensorComparator::allClose(t1, t2, atol, rtol))

#define ASSERT_BOUNDS_CLOSE(b1, b2, atol, rtol) \
    ASSERT_TRUE(test::TensorComparator::boundsClose(b1, b2, atol, rtol))

#endif // __TENSOR_COMPARATORS_H__
