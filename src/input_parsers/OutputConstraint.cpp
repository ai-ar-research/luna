/*********************                                                        */
/*! \file OutputConstraint.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "OutputConstraint.h"
#include "InputParserError.h"
#include "MStringf.h"

namespace NLR {

OutputConstraintSet::OutputConstraintSet()
    : _outputDim(0), _hasORDisjunction(false)
{
}

void OutputConstraintSet::setOutputDimension(unsigned dim)
{
    _outputDim = dim;
}

unsigned OutputConstraintSet::getOutputDimension() const
{
    return _outputDim;
}

void OutputConstraintSet::addConstraint(const OutputConstraint& constraint)
{
    _constraints.append(constraint);
}

unsigned OutputConstraintSet::getNumConstraints() const
{
    return _constraints.size();
}

bool OutputConstraintSet::hasConstraints() const
{
    return _constraints.size() > 0 || _orBranches.size() > 0;
}

void OutputConstraintSet::clear()
{
    _constraints.clear();
    _orBranches.clear();
    _hasORDisjunction = false;
}

void OutputConstraintSet::addORBranch(const Vector<OutputConstraint>& branch)
{
    _orBranches.append(branch);
    _hasORDisjunction = true;
}

bool OutputConstraintSet::hasORDisjunction() const
{
    return _hasORDisjunction;
}

unsigned OutputConstraintSet::getNumORBranches() const
{
    return _orBranches.size();
}

CMatrixResult OutputConstraintSet::toCMatrix() const
{
    CMatrixResult result;

    if (_outputDim == 0)
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                               "Output dimension must be set before calling toCMatrix()");
    }

    if (_hasORDisjunction)
    {
        if (_orBranches.size() == 0)
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                   "OR disjunction flag is set but no branches exist");
        }

        unsigned totalConstraints = 0;
        for (unsigned branchId = 0; branchId < _orBranches.size(); ++branchId)
        {
            totalConstraints += _orBranches[branchId].size();
            result.branchSizes.append(_orBranches[branchId].size());
        }

        if (totalConstraints == 0)
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                   "No constraints in OR branches to convert to C matrix");
        }

        result.C = torch::zeros({(long)totalConstraints, 1, (long)_outputDim}, torch::kFloat32);
        result.thresholds = torch::zeros({(long)totalConstraints}, torch::kFloat32);
        result.hasORBranches = true;
        unsigned rowIndex = 0;
        for (unsigned branchId = 0; branchId < _orBranches.size(); ++branchId)
        {
            const Vector<OutputConstraint>& branch = _orBranches[branchId];
            for (unsigned i = 0; i < branch.size(); ++i)
            {
                const OutputConstraint& constraint = branch[i];

                for (unsigned j = 0; j < constraint.terms.size(); ++j)
                {
                    const OutputTerm& term = constraint.terms[j];

                    if (term.outputIndex >= _outputDim)
                    {
                        throw InputParserError(InputParserError::VARIABLE_INDEX_OUT_OF_RANGE,
                                               Stringf("Output index %u exceeds output dimension %u",
                                                       term.outputIndex, _outputDim).ascii());
                    }

                    result.C[rowIndex][0][term.outputIndex] = static_cast<float>(term.coefficient);
                }

                result.thresholds[rowIndex] = static_cast<float>(constraint.threshold);

                result.branchMapping.append(branchId);

                ++rowIndex;
            }
        }
    }
    else
    {
        unsigned numConstraints = _constraints.size();
        if (numConstraints == 0)
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                   "No constraints to convert to C matrix");
        }

        result.C = torch::zeros({(long)numConstraints, 1, (long)_outputDim}, torch::kFloat32);
        result.thresholds = torch::zeros({(long)numConstraints}, torch::kFloat32);
        result.hasORBranches = false;

        for (unsigned i = 0; i < numConstraints; ++i)
        {
            const OutputConstraint& constraint = _constraints[i];

            // Fill in coefficients for this constraint
            for (unsigned j = 0; j < constraint.terms.size(); ++j)
            {
                const OutputTerm& term = constraint.terms[j];

                if (term.outputIndex >= _outputDim)
                {
                    throw InputParserError(InputParserError::VARIABLE_INDEX_OUT_OF_RANGE,
                                           Stringf("Output index %u exceeds output dimension %u",
                                                   term.outputIndex, _outputDim).ascii());
                }

                result.C[i][0][term.outputIndex] = static_cast<float>(term.coefficient);
            }

            // Set threshold
            result.thresholds[i] = static_cast<float>(constraint.threshold);
        }
    }

    return result;
}

torch::Tensor OutputConstraintSet::identityC(unsigned outputDim)
{
    torch::Tensor C = torch::zeros({(long)outputDim, 1, (long)outputDim}, torch::kFloat32);

    for (unsigned i = 0; i < outputDim; ++i)
    {
        C[i][0][i] = 1.0f;
    }

    return C;
}

Vector<BranchResult> OutputConstraintSet::evaluateORBranches(
    const torch::Tensor& lowerBounds,
    const torch::Tensor& upperBounds,
    const torch::Tensor& thresholds,
    const Vector<unsigned>& branchMapping,
    const Vector<unsigned>& branchSizes)
{
    Vector<BranchResult> results;

    if (branchSizes.size() == 0)
    {
        return results;
    }

    int totalRows = lowerBounds.size(0);
    if (upperBounds.size(0) != totalRows || thresholds.size(0) != totalRows ||
        (int)branchMapping.size() != totalRows)
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                               "Dimension mismatch in evaluateORBranches: all inputs must have same size");
    }

    Vector<Vector<unsigned>> branchRows;
    for (unsigned branchId = 0; branchId < branchSizes.size(); ++branchId)
    {
        branchRows.append(Vector<unsigned>());
    }

    for (int rowIndex = 0; rowIndex < totalRows; ++rowIndex)
    {
        unsigned branchId = branchMapping[rowIndex];
        if (branchId >= branchRows.size())
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                   Stringf("Branch ID %u out of range [0, %u)",
                                           branchId, branchRows.size()).ascii());
        }
        branchRows[branchId].append(static_cast<unsigned>(rowIndex));
    }

    for (unsigned branchId = 0; branchId < branchSizes.size(); ++branchId)
    {
        BranchResult branchResult;
        branchResult.branchId = branchId;
        branchResult.verified = false;
        branchResult.refuted = false;
        branchResult.rowIndices = branchRows[branchId];

        if (branchRows[branchId].size() == 0)
        {
            results.append(branchResult);
            continue;
        }

        // lb > threshold disproves the AND-conjunction for this branch
        bool branchDisproved = false;
        for (unsigned i = 0; i < branchRows[branchId].size(); ++i)
        {
            unsigned rowIndex = branchRows[branchId][i];

            float lowerDiff = lowerBounds[rowIndex].item<float>() - thresholds[rowIndex].item<float>();

            if (lowerDiff > 0.0f)
            {
                branchDisproved = true;
                break;
            }
        }
        branchResult.verified = branchDisproved;

        // ub <= threshold on all rows means counterexample is always satisfiable
        bool branchAlwaysSatisfiable = true;
        for (unsigned i = 0; i < branchRows[branchId].size(); ++i)
        {
            unsigned rowIndex = branchRows[branchId][i];

            float upperDiff = upperBounds[rowIndex].item<float>() - thresholds[rowIndex].item<float>();

            if (upperDiff > 0.0f)
            {
                branchAlwaysSatisfiable = false;
                break;
            }
        }
        branchResult.refuted = branchAlwaysSatisfiable;

        results.append(branchResult);
    }

    return results;
}

} // namespace NLR