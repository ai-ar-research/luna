#include "tensor_comparators.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace test
{

bool TensorComparator::allClose( const torch::Tensor &t1, const torch::Tensor &t2, double atol,
                                 double rtol )
{
    if ( t1.sizes() != t2.sizes() )
    {
        return false;
    }

    torch::Tensor diff = torch::abs( t1.flatten() - t2.flatten() );
    torch::Tensor absT2 = torch::abs( t2.flatten() );

    torch::Tensor maxDiff = torch::max( diff );
    torch::Tensor maxAbs = torch::max( absT2 );

    double maxDiffVal = maxDiff.item<double>();
    double threshold = atol + rtol * maxAbs.item<double>();

    return maxDiffVal <= threshold;
}

double TensorComparator::maxAbsDiff( const torch::Tensor &t1, const torch::Tensor &t2 )
{
    if ( t1.sizes() != t2.sizes() )
    {
        return std::numeric_limits<double>::infinity();
    }

    torch::Tensor diff = torch::abs( t1.flatten() - t2.flatten() );
    torch::Tensor maxDiff = torch::max( diff );
    return maxDiff.item<double>();
}

std::string TensorComparator::diffReport( const torch::Tensor &t1, const torch::Tensor &t2,
                                          double atol, double rtol )
{
    std::ostringstream oss;

    if ( t1.sizes() != t2.sizes() )
    {
        oss << "Shape mismatch: " << t1.sizes() << " vs " << t2.sizes();
        return oss.str();
    }

    torch::Tensor diff = torch::abs( t1.flatten() - t2.flatten() );
    torch::Tensor absT2 = torch::abs( t2.flatten() );
    torch::Tensor relDiff = diff / ( absT2 + atol );

    auto diffAccessor = diff.accessor<float, 1>();
    auto relAccessor = relDiff.accessor<float, 1>();
    auto t1Flat = t1.flatten();
    auto t2Flat = t2.flatten();
    auto t1Accessor = t1Flat.accessor<float, 1>();
    auto t2Accessor = t2Flat.accessor<float, 1>();

    int64_t n = diff.numel();
    int maxReport = 10;
    int reported = 0;

    double maxAbsDiff = 0.0;
    double maxRelDiff = 0.0;
    int64_t maxAbsIdx = 0;
    int64_t maxRelIdx = 0;

    for ( int64_t i = 0; i < n; ++i )
    {
        double absD = diffAccessor[i];
        double relD = relAccessor[i];
        if ( absD > maxAbsDiff )
        {
            maxAbsDiff = absD;
            maxAbsIdx = i;
        }
        if ( relD > maxRelDiff )
        {
            maxRelDiff = relD;
            maxRelIdx = i;
        }
    }

    oss << "Max absolute diff: " << std::scientific << std::setprecision( 6 ) << maxAbsDiff
        << " at index " << maxAbsIdx << "\n";
    oss << "Max relative diff: " << maxRelDiff << " at index " << maxRelIdx << "\n";
    oss << "First " << std::min( maxReport, (int)n ) << " mismatches:\n";

    for ( int64_t i = 0; i < n && reported < maxReport; ++i )
    {
        double threshold = atol + rtol * absT2[i].item<float>();
        if ( diffAccessor[i] > threshold )
        {
            oss << "  [" << i << "] " << t1Accessor[i] << " vs " << t2Accessor[i]
                << " (diff=" << diffAccessor[i] << ", rel=" << relAccessor[i] << ")\n";
            reported++;
        }
    }

    return oss.str();
}

bool TensorComparator::boundsClose( const BoundedTensor<torch::Tensor> &b1,
                                    const BoundedTensor<torch::Tensor> &b2, double atol,
                                    double rtol )
{
    return allClose( b1.lower(), b2.lower(), atol, rtol ) &&
           allClose( b1.upper(), b2.upper(), atol, rtol );
}

std::string TensorComparator::boundsDiffReport( const BoundedTensor<torch::Tensor> &b1,
                                                const BoundedTensor<torch::Tensor> &b2, double atol,
                                                double rtol )
{
    std::ostringstream oss;

    oss << "Lower bounds:\n" << diffReport( b1.lower(), b2.lower(), atol, rtol ) << "\n";
    oss << "Upper bounds:\n" << diffReport( b1.upper(), b2.upper(), atol, rtol );

    return oss.str();
}

} // namespace test
