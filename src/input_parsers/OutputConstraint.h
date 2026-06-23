/*********************                                                        */
/*! \file OutputConstraint.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __OutputConstraint_h__
#define __OutputConstraint_h__

#include "MString.h"
#include "Vector.h"

// avoid macro conflict with PyTorch
#ifdef Warning
#undef Warning
#endif
#ifdef LOG
#undef LOG
#endif

#include <torch/torch.h>

namespace NLR {

struct OutputTerm
{
    unsigned outputIndex;
    double coefficient;

    OutputTerm()
        : outputIndex(0), coefficient(1.0)
    {
    }

    OutputTerm(unsigned index, double coeff = 1.0)
        : outputIndex(index), coefficient(coeff)
    {
    }
};

// normalized form: sum_i(coeff_i * Y_i) <= threshold
struct OutputConstraint
{
    Vector<OutputTerm> terms;
    double threshold;

    OutputConstraint()
        : threshold(0.0)
    {
    }
};

struct CMatrixResult
{
    torch::Tensor C;
    torch::Tensor thresholds;
    Vector<unsigned> branchMapping;
    Vector<unsigned> branchSizes;
    bool hasORBranches;

    CMatrixResult()
        : hasORBranches(false)
    {
    }
};

struct BranchResult
{
    unsigned branchId;
    bool verified;
    bool refuted;
    Vector<unsigned> rowIndices;
};

class OutputConstraintSet
{
public:
    OutputConstraintSet();

    void setOutputDimension(unsigned dim);
    unsigned getOutputDimension() const;
    void addConstraint(const OutputConstraint& constraint);
    unsigned getNumConstraints() const;
    bool hasConstraints() const;
    void clear();
    void addORBranch(const Vector<OutputConstraint>& branch);
    bool hasORDisjunction() const;
    unsigned getNumORBranches() const;

    // negates coefficients/thresholds to match auto-LiRPA convention
    CMatrixResult toCMatrix() const;

    static Vector<BranchResult> evaluateORBranches(
        const torch::Tensor& lowerBounds,
        const torch::Tensor& upperBounds,
        const torch::Tensor& thresholds,
        const Vector<unsigned>& branchMapping,
        const Vector<unsigned>& branchSizes
    );

    static torch::Tensor identityC(unsigned outputDim);

private:
    unsigned _outputDim;
    Vector<OutputConstraint> _constraints;
    Vector<Vector<OutputConstraint>> _orBranches;
    bool _hasORDisjunction;
};

} // namespace NLR

#endif // __OutputConstraint_h__
