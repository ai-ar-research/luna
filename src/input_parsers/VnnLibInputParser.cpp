#include "VnnLibInputParser.h"
#include "File.h"
#include "InputParserError.h"
#include "MStringf.h"

#include <boost/regex.hpp>
#include <limits>

// Helper function to extract scalar values
double VnnLibInputParser::extractScalar(const String &token)
{
    std::string::size_type end;
    double value = std::stod(token.ascii(), &end);
    if (end != token.length())
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                              Stringf("'%s' is not a valid scalar", token.ascii()).ascii());
    }
    return value;
}

// Read VNN-LIB file and strip comments
String VnnLibInputParser::readVnnlibFile(const String &vnnlibFilePath)
{
//     std::cout << "[VnnLibInputParser] Reading file: " << vnnlibFilePath.ascii() << std::endl;

    if (!File::exists(vnnlibFilePath))
    {
        throw InputParserError(InputParserError::FILE_DOESNT_EXIST,
                              Stringf("VNN-LIB file not found: %s", vnnlibFilePath.ascii()).ascii());
    }

    File vnnlibFile(vnnlibFilePath);
    vnnlibFile.open(File::MODE_READ);

    String vnnlibContent;
    int lineCount = 0;

    try
    {
        while (true)
        {
            String line = vnnlibFile.readLine().trim();
            lineCount++;

            // Skip empty lines and comments
            if (line == "" || line.substring(0, 1) == ";")
            {
//                 std::cout << "[VnnLibInputParser] Skipping line " << lineCount
//                          << " (empty or comment)" << std::endl;
                continue;
            }

//             std::cout << "[VnnLibInputParser] Line " << lineCount << ": "
//                      << line.ascii() << std::endl;
            vnnlibContent += line;
        }
    }
    catch (const CommonError &e)
    {
        // READ_FAILED indicates end of file
        if (e.getCode() != CommonError::READ_FAILED)
            throw e;
    }

//     std::cout << "[VnnLibInputParser] Read " << lineCount << " lines total" << std::endl;
//     std::cout << "[VnnLibInputParser] Content length: " << vnnlibContent.length() << " characters" << std::endl;

    return vnnlibContent;
}

// Tokenize VNN-LIB content
Vector<String> VnnLibInputParser::tokenize(const String &vnnlibContent)
{
//     std::cout << "[VnnLibInputParser] Tokenizing content..." << std::endl;

    boost::regex re(R"(\(|\)|[\w\-\\.]+|<=|>=|\+|-|\*)");

    auto tokens_begin = boost::cregex_token_iterator(
        vnnlibContent.ascii(), vnnlibContent.ascii() + vnnlibContent.length(), re);
    auto tokens_end = boost::cregex_token_iterator();

    Vector<String> tokens;
    for (boost::cregex_token_iterator it = tokens_begin; it != tokens_end; ++it)
    {
        boost::csub_match match = *it;
        tokens.append(String(match.str().c_str()));
    }

//     std::cout << "[VnnLibInputParser] Generated " << tokens.size() << " tokens" << std::endl;
//     std::cout << "[VnnLibInputParser] First 20 tokens: ";
    // for (unsigned i = 0; i < tokens.size() && i < 20; ++i)
    // {
    //     std::cout << tokens[i].ascii() << " ";
    // }
    // std::cout << std::endl;

    return tokens;
}

// Check if variable name is an input variable (X_i format)
bool VnnLibInputParser::isInputVariable(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
        return false;

    return parts.front() == "X";
}

// Extract variable index from name (e.g., "X_0" -> 0, "Y_3" -> 3)
int VnnLibInputParser::extractVariableIndex(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
    {
        throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                              Stringf("Invalid variable name format: %s", varName.ascii()).ascii());
    }

    const String &indexStr = parts.back();
    for (unsigned i = 0; i < indexStr.length(); ++i)
    {
        if (!std::isdigit(indexStr[i]))
        {
            throw InputParserError(InputParserError::UNEXPECTED_INPUT,
                                  Stringf("Invalid variable index in: %s", varName.ascii()).ascii());
        }
    }

    return atoi(indexStr.ascii());
}

// Parse tokens and extract input bounds
void VnnLibInputParser::parseTokens(const Vector<String> &tokens,
                                    Vector<InputBoundInfo> &inputBounds)
{
    int index = 0;
    while ((unsigned)index < tokens.size())
    {
        if (tokens[index] == "(")
        {
            index = parseCommand(index + 1, tokens, inputBounds);
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
            {
                ++index;
            }
        }
        else
        {
            ++index;
        }
    }
}

// Parse a command (declare-const or assert)
int VnnLibInputParser::parseCommand(int index,
                                    const Vector<String> &tokens,
                                    Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &command = tokens[index];

    if (command == "declare-const")
    {
        return parseDeclareConst(index + 1, tokens, inputBounds);
    }
    else if (command == "assert")
    {
        return parseAssert(index + 1, tokens, inputBounds);
    }

    // Skip unknown commands - just find matching closing paren
    int depth = 1;
    ++index;
    while ((unsigned)index < tokens.size() && depth > 0)
    {
        if (tokens[index] == "(")
            ++depth;
        else if (tokens[index] == ")")
            --depth;
        ++index;
    }
    return index - 1;
}

// Parse declare-const to identify input variables
int VnnLibInputParser::parseDeclareConst(int index,
                                         const Vector<String> &tokens,
                                         Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index + 1 >= tokens.size())
        return index;

    const String &varName = tokens[index];
    const String &varType = tokens[index + 1];

//     std::cout << "[VnnLibInputParser] declare-const: " << varName.ascii()
//               << " " << varType.ascii() << std::endl;

    if (varType != "Real")
    {
        return index + 2;
    }

    // Only track input variables
    if (isInputVariable(varName))
    {
        int varIndex = extractVariableIndex(varName);
//         std::cout << "[VnnLibInputParser] Found input variable " << varName.ascii()
//                   << " (index " << varIndex << ")" << std::endl;

        // Ensure inputBounds vector is large enough
        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }
    }
    else
    {
//         std::cout << "[VnnLibInputParser] Skipping non-input variable: " << varName.ascii() << std::endl;
    }

    return index + 2;
}

// Parse assert command to extract bounds
int VnnLibInputParser::parseAssert(int index,
                                   const Vector<String> &tokens,
                                   Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    if (tokens[index] == "(")
    {
        return parseCondition(index + 1, tokens, inputBounds);
    }

    return index;
}

// Parse a condition (<= or >= or and)
int VnnLibInputParser::parseCondition(int index,
                                     const Vector<String> &tokens,
                                     Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    // Case-insensitive check for "and"
    if (op == "and" || op == "AND" || op == "And")
    {
        // Parse multiple conditions
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseCondition(index + 1, tokens, inputBounds);
            }
            else
            {
                ++index;
            }
        }
        return index + 1; // Skip closing )
    }
    else if (op == "or" || op == "OR" || op == "Or")
    {
        // OR may contain AND branches with input bounds (common VNN-LIB pattern:
        // (assert (or (and (>= X_0 -1) (<= X_0 1) (>= Y_0 100))))
        // Recurse into each branch to extract any input variable constraints.
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseCondition(index + 1, tokens, inputBounds);
            }
            else
            {
                ++index;
            }
        }
        return index + 1; // Skip closing )
    }
    else if (op == "<=" || op == ">=")
    {
        // Parse simple bound: (op var value) or (op value var)
        ++index;
        if ((unsigned)index + 1 >= tokens.size())
            return index;

        String token1 = tokens[index];
        String token2 = tokens[index + 1];

        String varName;
        String valueStr;
        bool varFirst = true;

        // Determine which token is variable and which is value
        if (isInputVariable(token1))
        {
            varName = token1;
            valueStr = token2;
            varFirst = true;
        }
        else if (isInputVariable(token2))
        {
            varName = token2;
            valueStr = token1;
            varFirst = false;
        }
        else
        {
            // Neither is an input variable, skip
            index += 2;
            // Find closing paren
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            return index + 1;
        }

        int varIndex = extractVariableIndex(varName);

        // Ensure inputBounds vector is large enough
        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }

        double value = extractScalar(valueStr);

        // Apply bound based on operator and order
        if (op == "<=")
        {
            if (varFirst)
            {
                // X <= value -> upper bound
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
//                 std::cout << "[VnnLibInputParser] Set upper bound: " << varName.ascii()
//                           << " <= " << value << std::endl;
            }
            else
            {
                // value <= X -> lower bound
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
//                 std::cout << "[VnnLibInputParser] Set lower bound: " << value
//                           << " <= " << varName.ascii() << std::endl;
            }
        }
        else // op == ">="
        {
            if (varFirst)
            {
                // X >= value -> lower bound
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
//                 std::cout << "[VnnLibInputParser] Set lower bound: " << varName.ascii()
//                           << " >= " << value << std::endl;
            }
            else
            {
                // value >= X -> upper bound
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
//                 std::cout << "[VnnLibInputParser] Set upper bound: " << value
//                           << " >= " << varName.ascii() << std::endl;
            }
        }

        index += 2;
        // Find closing paren
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    // Unknown operator, skip to closing paren
    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

// Main parsing entry point
BoundedTensor<torch::Tensor> VnnLibInputParser::parseInputBounds(const String &vnnlibFilePath,
                                                                  unsigned expectedInputSize)
{
//     std::cout << "\n[VnnLibInputParser] ========== Starting VNN-LIB Parsing ==========" << std::endl;
//     std::cout << "[VnnLibInputParser] Expected input size: " << expectedInputSize << std::endl;

    // Read and tokenize the file
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

//     std::cout << "[VnnLibInputParser] Parsing tokens to extract bounds..." << std::endl;

    // Parse tokens to extract bounds
    Vector<InputBoundInfo> inputBounds;
    parseTokens(tokens, inputBounds);

//     std::cout << "[VnnLibInputParser] Found " << inputBounds.size() << " input variables" << std::endl;

    // Ensure we have at least expectedInputSize entries
    while (inputBounds.size() < expectedInputSize)
    {
        inputBounds.append(InputBoundInfo());
    }

    // Warn if we found more inputs than expected
    if (inputBounds.size() > expectedInputSize)
    {
//         std::cout << "[VnnLibInputParser] WARNING: VNN-LIB file specifies " << inputBounds.size()
//                   << " input variables, but model expects " << expectedInputSize << std::endl;
    }

//     std::cout << "\n[VnnLibInputParser] Summary of extracted bounds:" << std::endl;
    // Debug loop - commented out to reduce output
    // for (unsigned i = 0; i < expectedInputSize; ++i)
    // {
    //     if (i < inputBounds.size())
    //     {
    //         std::cout << "[VnnLibInputParser]   X_" << i << ": ["
    //                   << inputBounds[i].lowerBound << ", "
    //                   << inputBounds[i].upperBound << "]";
    //         if (!inputBounds[i].hasLowerBound)
    //             std::cout << " (lower=default)";
    //         if (!inputBounds[i].hasUpperBound)
    //             std::cout << " (upper=default)";
    //         std::cout << std::endl;
    //     }
    // }

    // Convert to torch tensors - use double precision first for accurate parsing
    torch::Tensor lowerBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);
    torch::Tensor upperBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);

    for (unsigned i = 0; i < expectedInputSize; ++i)
    {
        if (i < inputBounds.size())
        {
            lowerBounds[i] = inputBounds[i].lowerBound;
            upperBounds[i] = inputBounds[i].upperBound;
        }
        else
        {
            // No bounds specified, use infinity
            lowerBounds[i] = -std::numeric_limits<double>::infinity();
            upperBounds[i] = std::numeric_limits<double>::infinity();
        }
    }

    // Now convert to float32 to match Python's precision
    // This ensures the conversion happens in the same way as Python
    lowerBounds = lowerBounds.to(torch::kFloat32);
    upperBounds = upperBounds.to(torch::kFloat32);

//     std::cout << "[VnnLibInputParser] ========== VNN-LIB Parsing Complete ==========" << std::endl << std::endl;

    return BoundedTensor<torch::Tensor>(lowerBounds, upperBounds);
}

// Check if variable name is an output variable (Y_i format)
bool VnnLibInputParser::isOutputVariable(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
        return false;

    return parts.front() == "Y";
}

// Check if a token is a scalar value (number)
bool VnnLibInputParser::isScalar(const String &token)
{
    if (token.length() == 0)
        return false;

    unsigned start = 0;
    // Handle optional negative sign
    if (token[0] == '-')
    {
        if (token.length() == 1)
            return false;
        start = 1;
    }

    bool hasDecimal = false;
    bool hasDigit = false;
    for (unsigned i = start; i < token.length(); ++i)
    {
        char c = token[i];
        if (c == '.')
        {
            if (hasDecimal)
                return false;
            hasDecimal = true;
        }
        else if (std::isdigit(c))
        {
            hasDigit = true;
        }
        else if (c == 'e' || c == 'E')
        {
            // Handle scientific notation: check rest is valid exponent
            if (!hasDigit || i + 1 >= token.length())
                return false;
            unsigned expStart = i + 1;
            if (token[expStart] == '-' || token[expStart] == '+')
            {
                ++expStart;
            }
            for (unsigned j = expStart; j < token.length(); ++j)
            {
                if (!std::isdigit(token[j]))
                    return false;
            }
            return true;
        }
        else
        {
            return false;
        }
    }
    return hasDigit;
}

// Parse tokens and extract output constraints
void VnnLibInputParser::parseOutputTokens(const Vector<String> &tokens,
                                          NLR::OutputConstraintSet &outputConstraints)
{
    int index = 0;
    while ((unsigned)index < tokens.size())
    {
        if (tokens[index] == "(")
        {
            index = parseOutputCommand(index + 1, tokens, outputConstraints);
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
            {
                ++index;
            }
        }
        else
        {
            ++index;
        }
    }
}

// Parse a command for output constraints
int VnnLibInputParser::parseOutputCommand(int index,
                                          const Vector<String> &tokens,
                                          NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &command = tokens[index];

    if (command == "assert")
    {
        return parseOutputAssert(index + 1, tokens, outputConstraints);
    }

    // Skip non-assert commands (like declare-const) - just find matching closing paren
    int depth = 1;
    ++index;
    while ((unsigned)index < tokens.size() && depth > 0)
    {
        if (tokens[index] == "(")
            ++depth;
        else if (tokens[index] == ")")
            --depth;
        ++index;
    }
    return index - 1;
}

// Parse an assert command for output constraints
int VnnLibInputParser::parseOutputAssert(int index,
                                         const Vector<String> &tokens,
                                         NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    if (tokens[index] == "(")
    {
        return parseOutputCondition(index + 1, tokens, outputConstraints);
    }

    return index;
}

// Parse a linear expression of output variables
// Handles: Y_i, (+ expr expr ...), (* coeff Y_i), (- Y_i Y_j), scalars
int VnnLibInputParser::parseLinearExpression(int index,
                                             const Vector<String> &tokens,
                                             Vector<NLR::OutputTerm> &terms,
                                             double &scalarSum)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &token = tokens[index];

    // Case 1: Simple output variable Y_i
    if (isOutputVariable(token))
    {
        int varIndex = extractVariableIndex(token);
        terms.append(NLR::OutputTerm(varIndex, 1.0));
        return index + 1;
    }

    // Case 2: Scalar value - accumulate it in scalarSum
    if (isScalar(token))
    {
        scalarSum += extractScalar(token);
        return index + 1;
    }

    // Case 3: Parenthesized expression
    if (token == "(")
    {
        ++index;
        if ((unsigned)index >= tokens.size())
            return index;

        const String &op = tokens[index];
        ++index;

        if (op == "+")
        {
            // Addition: (+ expr1 expr2 ...)
            // Parse all sub-expressions and collect their terms and scalars
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
            {
                index = parseLinearExpression(index, tokens, terms, scalarSum);
            }
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
                ++index;
            return index;
        }
        else if (op == "-")
        {
            // Subtraction: (- expr1 expr2)
            // First expression has positive coefficient
            Vector<NLR::OutputTerm> firstTerms;
            double firstScalar = 0.0;
            index = parseLinearExpression(index, tokens, firstTerms, firstScalar);
            for (unsigned i = 0; i < firstTerms.size(); ++i)
            {
                terms.append(firstTerms[i]);
            }
            scalarSum += firstScalar;

            // Second expression has negative coefficient
            if ((unsigned)index < tokens.size() && tokens[index] != ")")
            {
                Vector<NLR::OutputTerm> secondTerms;
                double secondScalar = 0.0;
                index = parseLinearExpression(index, tokens, secondTerms, secondScalar);
                for (unsigned i = 0; i < secondTerms.size(); ++i)
                {
                    NLR::OutputTerm negatedTerm = secondTerms[i];
                    negatedTerm.coefficient *= -1.0;
                    terms.append(negatedTerm);
                }
                // Subtract second scalar
                scalarSum -= secondScalar;
            }

            // Skip any remaining terms and closing paren
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
                ++index;
            return index;
        }
        else if (op == "*")
        {
            // Multiplication: (* coeff Y_i) or (* Y_i coeff)
            double coefficient = 1.0;
            int varIndex = -1;

            while ((unsigned)index < tokens.size() && tokens[index] != ")")
            {
                const String &subToken = tokens[index];

                if (isScalar(subToken))
                {
                    coefficient *= extractScalar(subToken);
                    ++index;
                }
                else if (isOutputVariable(subToken))
                {
                    varIndex = extractVariableIndex(subToken);
                    ++index;
                }
                else if (subToken == "(")
                {
                    // Nested expression within multiplication
                    Vector<NLR::OutputTerm> subTerms;
                    double subScalar = 0.0;
                    index = parseLinearExpression(index, tokens, subTerms, subScalar);
                    // Multiply all sub-terms by current coefficient
                    for (unsigned i = 0; i < subTerms.size(); ++i)
                    {
                        NLR::OutputTerm scaledTerm = subTerms[i];
                        scaledTerm.coefficient *= coefficient;
                        terms.append(scaledTerm);
                    }
                    // Multiply scalar by coefficient and add to scalarSum
                    scalarSum += subScalar * coefficient;
                    coefficient = 1.0; // Reset after applying
                    varIndex = -1; // Mark as already added
                }
                else
                {
                    ++index;
                }
            }

            if (varIndex >= 0)
            {
                terms.append(NLR::OutputTerm(varIndex, coefficient));
            }

            if ((unsigned)index < tokens.size() && tokens[index] == ")")
                ++index;
            return index;
        }
        else
        {
            // Unknown operator, skip to closing paren
            int depth = 1;
            while ((unsigned)index < tokens.size() && depth > 0)
            {
                if (tokens[index] == "(")
                    ++depth;
                else if (tokens[index] == ")")
                    --depth;
                ++index;
            }
            return index;
        }
    }

    // Unknown token, skip
    return index + 1;
}

// Parse an output condition and add constraints
int VnnLibInputParser::parseOutputCondition(int index,
                                            const Vector<String> &tokens,
                                            NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    // Case-insensitive check for "and"
    if (op == "and" || op == "AND" || op == "And")
    {
        // Parse multiple conditions within 'and'
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseOutputCondition(index + 1, tokens, outputConstraints);
            }
            else
            {
                ++index;
            }
        }
        return index + 1; // Skip closing )
    }
    // Case-insensitive check for "or"
    else if (op == "or" || op == "OR" || op == "Or")
    {
        // Parse OR disjunction: collect each branch separately
        // Each branch can be a single constraint or an AND conjunction
        ++index;
        
        unsigned outputDim = outputConstraints.getOutputDimension();
        Vector<Vector<NLR::OutputConstraint>> branches;
        
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                // Parse a single branch (can be AND or single constraint)
                Vector<NLR::OutputConstraint> branchConstraints;
                index = parseOutputBranch(index + 1, tokens, branchConstraints, outputDim);
                
                if (branchConstraints.size() > 0)
                {
                    branches.append(branchConstraints);
                }
            }
            else
            {
                ++index;
            }
        }
        
        // Add all branches as OR branches
        for (unsigned i = 0; i < branches.size(); ++i)
        {
            outputConstraints.addORBranch(branches[i]);
        }
        
        return index + 1;
    }
    else if (op == "<=" || op == ">=")
    {
        ++index;
        if ((unsigned)index >= tokens.size())
            return index;

        // Parse LHS and RHS of the comparison
        Vector<NLR::OutputTerm> lhsTerms;
        Vector<NLR::OutputTerm> rhsTerms;
        double lhsScalar = 0.0;
        double rhsScalar = 0.0;

        // Parse LHS
        if (tokens[index] == "(")
        {
            index = parseLinearExpression(index, tokens, lhsTerms, lhsScalar);
        }
        else if (isOutputVariable(tokens[index]))
        {
            int varIndex = extractVariableIndex(tokens[index]);
            lhsTerms.append(NLR::OutputTerm(varIndex, 1.0));
            ++index;
        }
        else if (isScalar(tokens[index]))
        {
            lhsScalar = extractScalar(tokens[index]);
            ++index;
        }
        else
        {
            ++index;
        }

        // Parse RHS
        if ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseLinearExpression(index, tokens, rhsTerms, rhsScalar);
            }
            else if (isOutputVariable(tokens[index]))
            {
                int varIndex = extractVariableIndex(tokens[index]);
                rhsTerms.append(NLR::OutputTerm(varIndex, 1.0));
                ++index;
            }
            else if (isScalar(tokens[index]))
            {
                rhsScalar = extractScalar(tokens[index]);
                ++index;
            }
            else
            {
                ++index;
            }
        }

        // Check if this constraint involves output variables
        bool hasOutputVar = (lhsTerms.size() > 0 || rhsTerms.size() > 0);

        if (hasOutputVar)
        {
            NLR::OutputConstraint constraint;

            // Normalize all constraints to: C*y <= threshold form
            // For >= constraints: negate coefficients and threshold to get -C*y <= -threshold
            // For <= constraints: keep as-is

            // Add LHS terms
            for (unsigned i = 0; i < lhsTerms.size(); ++i)
            {
                constraint.terms.append(lhsTerms[i]);
            }

            // Subtract RHS terms (negate coefficients)
            for (unsigned i = 0; i < rhsTerms.size(); ++i)
            {
                NLR::OutputTerm negated = rhsTerms[i];
                negated.coefficient *= -1.0;
                constraint.terms.append(negated);
            }

            // Compute threshold: rhsScalar - lhsScalar
            double threshold = rhsScalar - lhsScalar;

            // Normalize >= to <= form by negating coefficients and threshold
            if (op == ">=")
            {
                // C*y >= threshold -> -C*y <= -threshold
                for (unsigned i = 0; i < constraint.terms.size(); ++i)
                {
                    constraint.terms[i].coefficient *= -1.0;
                }
                threshold = -threshold;
            }
            // For <=, keep as-is (already in correct form)

            constraint.threshold = threshold;
            outputConstraints.addConstraint(constraint);
        }

        // Skip to closing paren
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    // Unknown operator, skip to closing paren
    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

// Parse a single OR branch (which may be an AND conjunction or a single constraint)
int VnnLibInputParser::parseOutputBranch(int index,
                                          const Vector<String> &tokens,
                                          Vector<NLR::OutputConstraint> &branchConstraints,
                                          unsigned outputDim)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    // Case-insensitive check for "and"
    if (op == "and" || op == "AND" || op == "And")
    {
        // Parse AND conjunction: collect all constraints in this branch
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                // Parse a single constraint within the AND
                index = parseOutputBranch(index + 1, tokens, branchConstraints, outputDim);
            }
            else
            {
                ++index;
            }
        }
        return index + 1; // Skip closing )
    }
    else if (op == "<=" || op == ">=")
    {
        // Parse a single constraint
        ++index;
        if ((unsigned)index >= tokens.size())
            return index;

        // Parse LHS and RHS of the comparison
        Vector<NLR::OutputTerm> lhsTerms;
        Vector<NLR::OutputTerm> rhsTerms;
        double lhsScalar = 0.0;
        double rhsScalar = 0.0;

        // Parse LHS
        if (tokens[index] == "(")
        {
            index = parseLinearExpression(index, tokens, lhsTerms, lhsScalar);
        }
        else if (isOutputVariable(tokens[index]))
        {
            int varIndex = extractVariableIndex(tokens[index]);
            lhsTerms.append(NLR::OutputTerm(varIndex, 1.0));
            ++index;
        }
        else if (isScalar(tokens[index]))
        {
            lhsScalar = extractScalar(tokens[index]);
            ++index;
        }
        else
        {
            ++index;
        }

        // Parse RHS
        if ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseLinearExpression(index, tokens, rhsTerms, rhsScalar);
            }
            else if (isOutputVariable(tokens[index]))
            {
                int varIndex = extractVariableIndex(tokens[index]);
                rhsTerms.append(NLR::OutputTerm(varIndex, 1.0));
                ++index;
            }
            else if (isScalar(tokens[index]))
            {
                rhsScalar = extractScalar(tokens[index]);
                ++index;
            }
            else
            {
                ++index;
            }
        }

        // Check if this constraint involves output variables
        bool hasOutputVar = (lhsTerms.size() > 0 || rhsTerms.size() > 0);

        if (hasOutputVar)
        {
            NLR::OutputConstraint constraint;

            // Normalize all constraints to: C*y <= threshold form
            // For >= constraints: negate coefficients and threshold to get -C*y <= -threshold
            // For <= constraints: keep as-is

            // Add LHS terms
            for (unsigned i = 0; i < lhsTerms.size(); ++i)
            {
                constraint.terms.append(lhsTerms[i]);
            }

            // Subtract RHS terms (negate coefficients)
            for (unsigned i = 0; i < rhsTerms.size(); ++i)
            {
                NLR::OutputTerm negated = rhsTerms[i];
                negated.coefficient *= -1.0;
                constraint.terms.append(negated);
            }

            // Compute threshold: rhsScalar - lhsScalar
            double threshold = rhsScalar - lhsScalar;

            // Normalize >= to <= form by negating coefficients and threshold
            if (op == ">=")
            {
                // C*y >= threshold -> -C*y <= -threshold
                for (unsigned i = 0; i < constraint.terms.size(); ++i)
                {
                    constraint.terms[i].coefficient *= -1.0;
                }
                threshold = -threshold;
            }
            // For <=, keep as-is (already in correct form)

            constraint.threshold = threshold;
            branchConstraints.append(constraint);
        }

        // Skip to closing paren
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    // Unknown operator, skip to closing paren
    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

// Parse output constraints from a VNN-LIB file
NLR::OutputConstraintSet VnnLibInputParser::parseOutputConstraints(const String &vnnlibFilePath,
                                                                    unsigned expectedOutputSize)
{
    // Read and tokenize the file
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

    // Parse tokens to extract output constraints
    NLR::OutputConstraintSet outputConstraints;
    outputConstraints.setOutputDimension(expectedOutputSize);
    parseOutputTokens(tokens, outputConstraints);

    return outputConstraints;
}

// Parse tokens for both input bounds and output constraints
void VnnLibInputParser::parseTokensBoth(const Vector<String> &tokens,
                                        Vector<InputBoundInfo> &inputBounds,
                                        NLR::OutputConstraintSet &outputConstraints)
{
    int index = 0;
    while ((unsigned)index < tokens.size())
    {
        if (tokens[index] == "(")
        {
            index = parseCommandBoth(index + 1, tokens, inputBounds, outputConstraints);
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
            {
                ++index;
            }
        }
        else
        {
            ++index;
        }
    }
}

// Parse a command for both inputs and outputs
int VnnLibInputParser::parseCommandBoth(int index,
                                        const Vector<String> &tokens,
                                        Vector<InputBoundInfo> &inputBounds,
                                        NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &command = tokens[index];

    if (command == "declare-const")
    {
        return parseDeclareConst(index + 1, tokens, inputBounds);
    }
    else if (command == "assert")
    {
        return parseAssertBoth(index + 1, tokens, inputBounds, outputConstraints);
    }

    // Skip unknown commands
    int depth = 1;
    ++index;
    while ((unsigned)index < tokens.size() && depth > 0)
    {
        if (tokens[index] == "(")
            ++depth;
        else if (tokens[index] == ")")
            --depth;
        ++index;
    }
    return index - 1;
}

// Parse an assert command for both inputs and outputs
int VnnLibInputParser::parseAssertBoth(int index,
                                       const Vector<String> &tokens,
                                       Vector<InputBoundInfo> &inputBounds,
                                       NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    if (tokens[index] == "(")
    {
        return parseConditionBoth(index + 1, tokens, inputBounds, outputConstraints);
    }

    return index;
}

// Parse a condition for both inputs and outputs
int VnnLibInputParser::parseConditionBoth(int index,
                                          const Vector<String> &tokens,
                                          Vector<InputBoundInfo> &inputBounds,
                                          NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    // Case-insensitive check for "and"
    if (op == "and" || op == "AND" || op == "And")
    {
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
                index = parseConditionBoth(index + 1, tokens, inputBounds, outputConstraints);
            }
            else
            {
                ++index;
            }
        }
        return index + 1;
    }
    // Case-insensitive check for "or"
    else if (op == "or" || op == "OR" || op == "Or")
    {
        // Check if this OR involves output variables by peeking ahead
        int peekIndex = index + 1;
        bool hasOutputVar = false;
        int depth = 1;
        while ((unsigned)peekIndex < tokens.size() && depth > 0)
        {
            if (tokens[peekIndex] == "(")
                ++depth;
            else if (tokens[peekIndex] == ")")
                --depth;
            else if (isOutputVariable(tokens[peekIndex]))
                hasOutputVar = true;
            ++peekIndex;
        }

        if (hasOutputVar)
        {
            // This OR involves output variables - parse both input bounds and output constraints.
            // VNN-LIB commonly embeds input bounds alongside output constraints inside
            // (or (and (input constraints) (output constraints))) blocks.
            ++index;

            unsigned outputDim = outputConstraints.getOutputDimension();
            Vector<Vector<NLR::OutputConstraint>> branches;

            while ((unsigned)index < tokens.size() && tokens[index] != ")")
            {
                if (tokens[index] == "(")
                {
                    ++index;
                    if ((unsigned)index >= tokens.size())
                        break;

                    const String &branchOp = tokens[index];

                    if (branchOp == "and" || branchOp == "AND" || branchOp == "And")
                    {
                        // AND branch: parse each sub-condition, routing input
                        // constraints to inputBounds and output constraints to branchConstraints
                        ++index;
                        Vector<NLR::OutputConstraint> branchConstraints;

                        while ((unsigned)index < tokens.size() && tokens[index] != ")")
                        {
                            if (tokens[index] == "(")
                            {
                                // Peek to see if this sub-condition is input or output
                                int subPeek = index + 1;
                                bool subHasInput = false;
                                bool subHasOutput = false;
                                int subDepth = 1;
                                while ((unsigned)subPeek < tokens.size() && subDepth > 0)
                                {
                                    if (tokens[subPeek] == "(") ++subDepth;
                                    else if (tokens[subPeek] == ")") --subDepth;
                                    else if (isInputVariable(tokens[subPeek])) subHasInput = true;
                                    else if (isOutputVariable(tokens[subPeek])) subHasOutput = true;
                                    ++subPeek;
                                }

                                if (subHasInput && !subHasOutput)
                                {
                                    // Input constraint - apply to inputBounds
                                    index = parseCondition(index + 1, tokens, inputBounds);
                                }
                                else if (subHasOutput)
                                {
                                    // Output constraint - add to branch
                                    index = parseOutputBranch(index + 1, tokens, branchConstraints, outputDim);
                                }
                                else
                                {
                                    // Skip unknown
                                    int skipDepth = 1;
                                    ++index;
                                    while ((unsigned)index < tokens.size() && skipDepth > 0)
                                    {
                                        if (tokens[index] == "(") ++skipDepth;
                                        else if (tokens[index] == ")") --skipDepth;
                                        ++index;
                                    }
                                }
                            }
                            else
                            {
                                ++index;
                            }
                        }

                        if (branchConstraints.size() > 0)
                        {
                            branches.append(branchConstraints);
                        }
                        if ((unsigned)index < tokens.size() && tokens[index] == ")")
                            ++index;
                    }
                    else
                    {
                        // Single constraint (not AND) - try as output branch
                        Vector<NLR::OutputConstraint> branchConstraints;
                        index = parseOutputBranch(index, tokens, branchConstraints, outputDim);
                        if (branchConstraints.size() > 0)
                        {
                            branches.append(branchConstraints);
                        }
                    }
                }
                else
                {
                    ++index;
                }
            }

            // Add all branches as OR branches
            for (unsigned i = 0; i < branches.size(); ++i)
            {
                outputConstraints.addORBranch(branches[i]);
            }

            return index + 1;
        }
        else
        {
            // This OR only involves input variables - handle as before
            ++index;
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
            {
                if (tokens[index] == "(")
                {
                    index = parseConditionBoth(index + 1, tokens, inputBounds, outputConstraints);
                }
                else
                {
                    ++index;
                }
            }
            return index + 1;
        }
    }
    else if (op == "<=" || op == ">=")
    {
        // Look ahead to determine if this involves input or output variables
        int peekIndex = index + 1;
        bool hasInputVar = false;
        bool hasOutputVar = false;

        // Quick scan to check variable types
        int depth = 1;
        while ((unsigned)peekIndex < tokens.size() && depth > 0)
        {
            if (tokens[peekIndex] == "(")
                ++depth;
            else if (tokens[peekIndex] == ")")
                --depth;
            else if (isInputVariable(tokens[peekIndex]))
                hasInputVar = true;
            else if (isOutputVariable(tokens[peekIndex]))
                hasOutputVar = true;
            ++peekIndex;
        }

        // Route to appropriate handler
        if (hasInputVar && !hasOutputVar)
        {
            // Input-only constraint - use input parsing
            return parseCondition(index, tokens, inputBounds);
        }
        else if (hasOutputVar && !hasInputVar)
        {
            // Output-only constraint - use output parsing
            // Create a temporary constraint set and add to the main one
            return parseOutputCondition(index, tokens, outputConstraints);
        }
        else
        {
            // Mixed or neither - skip to closing paren
            ++index;
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            return index + 1;
        }
    }

    // Unknown operator
    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

// Parse both input bounds and output constraints from a VNN-LIB file
std::pair<BoundedTensor<torch::Tensor>, NLR::OutputConstraintSet>
VnnLibInputParser::parseInputAndOutputConstraints(const String &vnnlibFilePath,
                                                  unsigned expectedInputSize,
                                                  unsigned expectedOutputSize)
{
    // Read and tokenize the file once
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

    // Parse both simultaneously
    Vector<InputBoundInfo> inputBounds;
    NLR::OutputConstraintSet outputConstraints;
    outputConstraints.setOutputDimension(expectedOutputSize);

    parseTokensBoth(tokens, inputBounds, outputConstraints);

    // Convert input bounds to tensors
    while (inputBounds.size() < expectedInputSize)
    {
        inputBounds.append(InputBoundInfo());
    }

    torch::Tensor lowerBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);
    torch::Tensor upperBounds = torch::zeros({(long)expectedInputSize}, torch::kFloat64);

    for (unsigned i = 0; i < expectedInputSize; ++i)
    {
        if (i < inputBounds.size())
        {
            lowerBounds[i] = inputBounds[i].lowerBound;
            upperBounds[i] = inputBounds[i].upperBound;
        }
        else
        {
            lowerBounds[i] = -std::numeric_limits<double>::infinity();
            upperBounds[i] = std::numeric_limits<double>::infinity();
        }
    }

    lowerBounds = lowerBounds.to(torch::kFloat32);
    upperBounds = upperBounds.to(torch::kFloat32);

    BoundedTensor<torch::Tensor> inputBoundsTensor(lowerBounds, upperBounds);

    return std::make_pair(inputBoundsTensor, outputConstraints);
}
