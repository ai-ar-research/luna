#include "src/input_parsers/OnnxToTorch.h"
#include "src/engine/TorchModel.h"
#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/input_parsers/VnnLibInputParser.h"
#include "src/configuration/LunaConfiguration.h"
#include "src/input_parsers/OutputConstraint.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>

// Helper function to print C matrix
static void printCMatrix( const torch::Tensor &C, const torch::Tensor &thresholds )
{
    std::cout << "\n=== C Matrix ===" << std::endl;
    std::cout << "Shape: [" << C.size( 0 ) << ", " << C.size( 1 ) << ", " << C.size( 2 ) << "]"
              << std::endl;
    std::cout << "Number of constraints: " << C.size( 0 ) << std::endl;
    std::cout << std::fixed << std::setprecision( 6 );

    // C has shape (num_constraints, 1, output_dim)
    // All constraints are normalized to C*y <= threshold form
    for ( int i = 0; i < C.size( 0 ); ++i )
    {
        std::cout << "\nConstraint " << i << ":" << std::endl;
        std::cout << "  C[" << i << "] = [";

        // Print the constraint coefficients
        auto constraintRow = C[i][0]; // Shape: (output_dim,)
        for ( int j = 0; j < constraintRow.size( 0 ); ++j )
        {
            if ( j > 0 )
                std::cout << ", ";
            std::cout << constraintRow[j].item<float>();
        }
        std::cout << "]" << std::endl;

        // Print threshold (all constraints are C*y <= threshold)
        std::cout << "  Threshold: " << thresholds[i].item<float>() << std::endl;
        std::cout << "  Constraint: C*y <= threshold" << std::endl;
    }
    std::cout << std::defaultfloat;
}

// Helper function to print bounds with optional rounding
static void printBounds( const torch::Tensor &lower, const torch::Tensor &upper, bool round = true )
{
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if ( lb.dim() == 0 )
        lb = lb.unsqueeze( 0 );
    if ( ub.dim() == 0 )
        ub = ub.unsqueeze( 0 );

    std::cout << "Output Bounds:" << std::endl;

    if ( round )
    {
        // Print with 6 decimal places when rounding
        std::cout << std::fixed << std::setprecision( 6 );
    }
    else
    {
        // Print with full precision (typically 15 significant digits for double)
        std::cout << std::scientific << std::setprecision( 15 );
    }

    for ( int i = 0; i < lb.size( 0 ); ++i )
    {
        if ( i > 0 )
            std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if ( l.dim() > 0 )
            l = l.flatten()[0];
        if ( u.dim() > 0 )
            u = u.flatten()[0];
        // Use double for consistency with Python's float32 precision handling
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
    std::cout << std::defaultfloat;
}

// Helper function to perform verification (for regular constraints)
// All constraints are normalized to C*y <= threshold form
static void performVerification( const torch::Tensor &thresholds, const torch::Tensor &lowerBounds,
                                 const torch::Tensor &upperBounds )
{
    std::cout << "\n=== Verification Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision( 6 );

    // When using a specification matrix C, the bounds returned are for C * output
    // So lowerBounds and upperBounds have shape (num_constraints,)
    // All constraints are normalized to C*y <= threshold form

    bool allVerified = true;

    // Check bounds shape matches number of constraints
    int numConstraints = thresholds.size( 0 );
    if ( lowerBounds.size( 0 ) != numConstraints || upperBounds.size( 0 ) != numConstraints )
    {
        std::cerr << "ERROR: Bounds shape mismatch. Expected " << numConstraints
                  << " constraints, but got bounds with shape [" << lowerBounds.size( 0 ) << "]"
                  << std::endl;
        return;
    }

    for ( int i = 0; i < numConstraints; ++i )
    {
        float threshold = thresholds[i].item<float>();
        float lowerBound = lowerBounds[i].item<float>();
        float upperBound = upperBounds[i].item<float>();

        // Constraint is C*y <= threshold
        // Verified if upperBound <= threshold (worst case still satisfies)
        // Refuted if lowerBound > threshold (even best case violates)
        bool satisfied = ( upperBound <= threshold );

        std::cout << "Constraint " << i << ": C*y <= " << threshold << std::endl;
        std::cout << "  Lower bound: " << lowerBound << ", Upper bound: " << upperBound
                  << std::endl;
        std::cout << "  Status: " << ( satisfied ? "VERIFIED" : "NOT VERIFIED" ) << std::endl;

        if ( !satisfied )
        {
            allVerified = false;
        }
    }

    std::cout << "\nOverall Verification: " << ( allVerified ? "VERIFIED" : "NOT VERIFIED" )
              << std::endl;
    std::cout << std::defaultfloat;
}

// Helper function to perform verification for OR branches
// All constraints are normalized to C*y <= threshold form
static void performORBranchVerification( const torch::Tensor &thresholds,
                                         const torch::Tensor &lowerBounds,
                                         const torch::Tensor &upperBounds,
                                         const Vector<unsigned> &branchMapping,
                                         const Vector<unsigned> &branchSizes )
{
    std::cout << "\n=== OR Branch Verification Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision( 6 );

    // Evaluate OR branches (all constraints are C*y <= threshold)
    Vector<NLR::BranchResult> branchResults = NLR::OutputConstraintSet::evaluateORBranches(
        lowerBounds, upperBounds, thresholds, branchMapping, branchSizes );

    bool anyBranchVerified = false;
    bool allBranchesRefuted = true;

    for ( unsigned i = 0; i < branchResults.size(); ++i )
    {
        const NLR::BranchResult &result = branchResults[i];

        std::cout << "\nBranch " << result.branchId << ":" << std::endl;
        std::cout << "  Row indices: [";
        for ( unsigned j = 0; j < result.rowIndices.size(); ++j )
        {
            if ( j > 0 )
                std::cout << ", ";
            std::cout << result.rowIndices[j];
        }
        std::cout << "]" << std::endl;

        // Print details for each row in this branch
        for ( unsigned j = 0; j < result.rowIndices.size(); ++j )
        {
            unsigned rowIndex = result.rowIndices[j];
            float threshold = thresholds[rowIndex].item<float>();
            float lowerBound = lowerBounds[rowIndex].item<float>();
            float upperBound = upperBounds[rowIndex].item<float>();

            // Constraint is C*y <= threshold
            // Verified if upperBound <= threshold
            // Refuted if lowerBound > threshold
            bool verified = ( upperBound <= threshold );
            bool refuted = ( lowerBound > threshold );

            std::cout << "    Row " << rowIndex << ": C*y <= " << threshold << std::endl;
            std::cout << "      Lower: " << lowerBound << ", Upper: " << upperBound;
            if ( verified )
            {
                std::cout << " ✓ (verified)";
            }
            else if ( refuted )
            {
                std::cout << " ✗ (refuted)";
            }
            else
            {
                std::cout << " ? (unknown)";
            }
            std::cout << std::endl;
        }

        std::cout << "  Branch status: ";
        if ( result.verified )
        {
            std::cout << "VERIFIED (all rows have upperBound <= threshold)";
            anyBranchVerified = true;
            allBranchesRefuted = false;
        }
        else if ( result.refuted )
        {
            std::cout << "REFUTED (at least one row has lowerBound > threshold)";
            // Keep allBranchesRefuted as true only if ALL branches are refuted
        }
        else
        {
            std::cout << "UNKNOWN (neither verified nor refuted)";
            allBranchesRefuted = false;
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Overall OR Verification ===" << std::endl;
    if ( anyBranchVerified )
    {
        std::cout << "VERIFIED: At least one branch is verified" << std::endl;
    }
    else if ( allBranchesRefuted && branchResults.size() > 0 )
    {
        std::cout << "REFUTED: All branches are refuted" << std::endl;
    }
    else
    {
        std::cout << "UNKNOWN: Property cannot be verified or refuted" << std::endl;
    }
    std::cout << std::defaultfloat;
}

int main( int argc, char *argv[] )
{
    // Parse command line arguments for configuration
    LunaConfiguration::parseArgs( argc, argv );
    (void)argc;
    (void)argv;

    try
    {
        // Force single-threaded execution for numerical determinism
        at::set_num_threads( 1 );
        at::set_num_interop_threads( 1 );
        std::cout << "Set threads to 1 for deterministic results\n\n";
        // Configuration flags
        bool round = true;

        // Default file paths (can be modified or passed as arguments)
        std::string onnxFilePath = "../resources/onnx/spec_test.onnx";
        std::string vnnlibFilePath = "../resources/properties/spec_test.vnnlib";

        unsigned iterations = 20;

        std::cout << "Parsing ONNX file: " << onnxFilePath << std::endl;
        std::cout << "Parsing VNN-LIB file: " << vnnlibFilePath << std::endl;

        // Step 1: Create TorchModel from ONNX file
        std::shared_ptr<NLR::TorchModel> torchModel =
            std::make_shared<NLR::TorchModel>( String( onnxFilePath.c_str() ) );

        std::cout << "TorchModel created successfully" << std::endl;
        std::cout << "Input size: " << torchModel->getInputSize() << std::endl;
        std::cout << "Output size: " << torchModel->getOutputSize() << std::endl;
        std::cout << "Number of nodes: " << torchModel->getNumNodes() << std::endl;

        // Step 2: Parse input bounds from VNN-LIB file
        BoundedTensor<torch::Tensor> inputBounds = VnnLibInputParser::parseInputBounds(
            String( vnnlibFilePath.c_str() ), torchModel->getInputSize() );

        torch::Tensor lowerBounds = inputBounds.lower();
        torch::Tensor upperBounds = inputBounds.upper();

        // Convert to float32 if needed
        if ( lowerBounds.dtype() != torch::kFloat32 )
        {
            lowerBounds = lowerBounds.to( torch::kFloat32 );
            upperBounds = upperBounds.to( torch::kFloat32 );
            inputBounds = BoundedTensor<torch::Tensor>( lowerBounds, upperBounds );
        }

        torchModel->setInputBounds( inputBounds );

        std::cout << "\nInput bounds loaded from VNN-LIB file:" << std::endl;
        std::cout << "  Input bounds: " << lowerBounds.size( 0 ) << " variables bounded in ["
                  << lowerBounds.min().item<double>() << ", " << upperBounds.max().item<double>()
                  << "]" << std::endl;

        // Step 3: Parse output constraints from VNN-LIB file
        std::cout << "\nParsing output constraints from VNN-LIB file..." << std::endl;
        NLR::OutputConstraintSet outputConstraints = VnnLibInputParser::parseOutputConstraints(
            String( vnnlibFilePath.c_str() ), torchModel->getOutputSize() );

        if ( !outputConstraints.hasConstraints() )
        {
            std::cerr << "ERROR: No output constraints found in VNN-LIB file!" << std::endl;
            return 1;
        }

        std::cout << "Found " << outputConstraints.getNumConstraints() << " output constraints"
                  << std::endl;
        if ( outputConstraints.hasORDisjunction() )
        {
            std::cout << "Property contains OR disjunction with "
                      << outputConstraints.getNumORBranches() << " branches" << std::endl;
        }

        // Step 4: Convert output constraints to C matrix
        NLR::CMatrixResult cMatrixResult = outputConstraints.toCMatrix();
        torch::Tensor C = cMatrixResult.C;
        torch::Tensor thresholds = cMatrixResult.thresholds;

        // Step 5: Output the C matrix
        printCMatrix( C, thresholds );

        if ( cMatrixResult.hasORBranches )
        {
            std::cout << "\nOR Branch Structure:" << std::endl;
            std::cout << "  Total rows: " << cMatrixResult.branchMapping.size() << std::endl;
            std::cout << "  Number of branches: " << cMatrixResult.branchSizes.size() << std::endl;
            for ( unsigned i = 0; i < cMatrixResult.branchSizes.size(); ++i )
            {
                std::cout << "  Branch " << i << ": " << cMatrixResult.branchSizes[i]
                          << " constraints" << std::endl;
            }
        }

        // Step 6: Run verification with CROWN
        std::cout << "\nRunning CROWN analysis with specification matrix..." << std::endl;
        BoundedTensor<torch::Tensor> crownResult =
            torchModel->compute_bounds( inputBounds,
                                        &C, // Specification matrix
                                        LunaConfiguration::AnalysisMethod::CROWN,
                                        true, // compute lower bounds
                                        true  // compute upper bounds
            );

        std::cout << "\n=== CROWN Results (with specification) ===" << std::endl;
        if ( crownResult.lower().defined() && crownResult.upper().defined() )
        {
            printBounds( crownResult.lower(), crownResult.upper(), round );

            // Perform verification - use OR branch evaluation if OR branches exist
            if ( cMatrixResult.hasORBranches )
            {
                performORBranchVerification( thresholds, crownResult.lower(), crownResult.upper(),
                                             cMatrixResult.branchMapping,
                                             cMatrixResult.branchSizes );
            }
            else
            {
                performVerification( thresholds, crownResult.lower(), crownResult.upper() );
            }
        }
        else
        {
            std::cout << "Bounds are undefined" << std::endl;
        }

        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
