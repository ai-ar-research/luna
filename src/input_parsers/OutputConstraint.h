#ifndef __OutputConstraint_h__
#define __OutputConstraint_h__

#include "MString.h"
#include "Vector.h"

// Undefine macros to avoid conflicts with PyTorch
#ifdef Warning
#undef Warning
#endif
#ifdef LOG
#undef LOG
#endif

#include <torch/torch.h>

namespace NLR {

/**
 * Represents a single term in a linear combination of output variables.
 * E.g., in "2*Y_0 - Y_1", one term would be {outputIndex=0, coefficient=2.0}
 */
struct OutputTerm
{
    unsigned outputIndex;  // Y_i index
    double coefficient;    // multiplier (default 1.0 for Y_i, can be negative)

    OutputTerm()
        : outputIndex(0), coefficient(1.0)
    {
    }

    OutputTerm(unsigned index, double coeff = 1.0)
        : outputIndex(index), coefficient(coeff)
    {
    }
};

/**
 * Represents a single output constraint in the normalized form:
 *   sum_i(coefficient_i * Y_i) <= threshold
 *
 * All constraints are normalized to this form during parsing.
 * For >= constraints from VNN-LIB, coefficients and threshold are negated.
 *
 * Examples:
 *   - Y_0 <= 0.5              -> terms=[{0, 1.0}], threshold=0.5
 *   - Y_0 - Y_1 <= 0          -> terms=[{0, 1.0}, {1, -1.0}], threshold=0
 *   - Y_2 <= 3.0              -> terms=[{2, 1.0}], threshold=3.0
 *   - Y_0 >= 0.5 (normalized) -> terms=[{0, -1.0}], threshold=-0.5
 */
struct OutputConstraint
{
    Vector<OutputTerm> terms;   // Linear combination terms
    double threshold;           // RHS value (threshold for <= constraint)

    OutputConstraint()
        : threshold(0.0)
    {
    }
};

/**
 * Result structure from toCMatrix() conversion.
 * Contains everything needed for CROWN
 */
struct CMatrixResult
{
    torch::Tensor C;                    // Shape: (total_rows, 1, output_dim)
    torch::Tensor thresholds;           // Shape: (total_rows,)
    Vector<unsigned> branchMapping;     // row_index -> branch_id (maps each constraint row to its OR branch)
    Vector<unsigned> branchSizes;       // branch_id -> num_constraints_in_branch
    bool hasORBranches;                 // whether OR branches exist

    CMatrixResult()
        : hasORBranches(false)
    {
    }
};

/**
 * Result of evaluating an OR branch
 */
struct BranchResult
{
    unsigned branchId;          // Branch identifier
    bool verified;              // true if branch is verified (all rows have certified lower bound >= 0)
    bool refuted;               // true if branch is refuted (at least one row has certified upper bound < 0)
    Vector<unsigned> rowIndices; // Row indices belonging to this branch
};

/**
 * Manages a collection of output constraints and converts them to
 * specification matrices for CROWN analysis.
 *
 * Usage:
 *   OutputConstraintSet constraints;
 *   constraints.setOutputDimension(10);
 *   constraints.addConstraint(...);
 *   CMatrixResult result = constraints.toCMatrix();
 */
class OutputConstraintSet
{
public:
    OutputConstraintSet();

    /**
     * Set the expected output dimension of the network.
     * Must be called before toCMatrix().
     */
    void setOutputDimension(unsigned dim);

    /**
     * Get the configured output dimension.
     */
    unsigned getOutputDimension() const;

    /**
     * Add an output constraint to the set.
     */
    void addConstraint(const OutputConstraint& constraint);

    /**
     * Get the number of constraints added.
     */
    unsigned getNumConstraints() const;

    /**
     * Check if any constraints have been added.
     */
    bool hasConstraints() const;

    /**
     * Clear all constraints.
     */
    void clear();

    /**
     * Add an OR branch (a collection of constraints that form one branch of an OR disjunction).
     * All constraints in a branch are ANDed together.
     */
    void addORBranch(const Vector<OutputConstraint>& branch);

    /**
     * Check if this constraint set has OR disjunctions.
     */
    bool hasORDisjunction() const;

    /**
     * Get the number of OR branches.
     */
    unsigned getNumORBranches() const;

    /**
     * Convert all constraints to a negated C matrix for CROWN backward propagation.
     *
     * VNN-LIB constraints are in the form C_orig*y <= d. This method negates both the
     * coefficients and thresholds to match auto-LiRPA convention: -C_orig*y >= -d.
     * With this convention, a positive lower bound means the property is satisfied (safe).
     *
     * Examples (original constraint → stored C row, threshold):
     *   - Y_0 - Y_1 <= 0.5: C row = [-1, 1, 0, ...], threshold = -0.5
     *   - Y_2 <= 3.0: C row = [0, 0, -1, ...], threshold = -3.0
     *
     * When OR branches exist, all constraints from all branches are concatenated into one
     * batched C matrix, and branch mapping information is populated.
     *
     * The returned C matrix has shape (total_rows, 1, output_dim) to match
     * expected format for batch processing.
     *
     * @return CMatrixResult containing negated C matrix, negated thresholds, and branch metadata
     */
    CMatrixResult toCMatrix() const;

    /**
     * Evaluate OR branches from batched bounds.
     *
     * All constraints are in negated auto-LiRPA convention: -C*y >= -d.
     * Bounds and thresholds are both in this negated space.
     * For each branch:
     *   - Verified: all rows have lowerBound >= threshold (safety margin is non-negative)
     *   - Refuted: at least one row has upperBound < threshold (even best case fails)
     *
     * @param lowerBounds Lower bounds for each constraint row (shape: total_rows)
     * @param upperBounds Upper bounds for each constraint row (shape: total_rows)
     * @param thresholds Thresholds for each constraint row (shape: total_rows)
     * @param branchMapping Mapping from row index to branch ID (length: total_rows)
     * @param branchSizes Number of constraints in each branch (length: num_branches)
     * @return Vector of BranchResult, one per branch
     */
    static Vector<BranchResult> evaluateORBranches(
        const torch::Tensor& lowerBounds,
        const torch::Tensor& upperBounds,
        const torch::Tensor& thresholds,
        const Vector<unsigned>& branchMapping,
        const Vector<unsigned>& branchSizes
    );

    /**
     * Create an identity C matrix (for verifying individual output bounds).
     * Shape: (output_dim, 1, output_dim)
     *
     * This is the default specification when no output constraints are provided,
     * corresponding to computing bounds on each output independently.
     *
     * @param outputDim Number of output dimensions
     * @return Identity matrix in the CROWN specification format
     */
    static torch::Tensor identityC(unsigned outputDim);

private:
    unsigned _outputDim;
    Vector<OutputConstraint> _constraints;           // Regular constraints (when no OR disjunction)
    Vector<Vector<OutputConstraint>> _orBranches;    // OR branches (each branch is a vector of constraints)
    bool _hasORDisjunction;                          // Flag indicating if OR disjunction exists
};

} // namespace NLR

#endif // __OutputConstraint_h__
