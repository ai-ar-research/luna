/*********************                                                        */
/*! \file VnnLibInputParser.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "VnnLibInputParser.h"
#include "File.h"
#include "InputParserError.h"
#include "MStringf.h"

#include <boost/regex.hpp>
#include <limits>

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

String VnnLibInputParser::readVnnlibFile(const String &vnnlibFilePath)
{
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

            if (line == "" || line.substring(0, 1) == ";")
                continue;

            vnnlibContent += line;
        }
    }
    catch (const CommonError &e)
    {
        // READ_FAILED indicates end of file
        if (e.getCode() != CommonError::READ_FAILED)
            throw e;
    }

    return vnnlibContent;
}

Vector<String> VnnLibInputParser::tokenize(const String &vnnlibContent)
{
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

    return tokens;
}

bool VnnLibInputParser::isInputVariable(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
        return false;

    return parts.front() == "X";
}

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

int VnnLibInputParser::parseDeclareConst(int index,
                                         const Vector<String> &tokens,
                                         Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index + 1 >= tokens.size())
        return index;

    const String &varName = tokens[index];
    const String &varType = tokens[index + 1];

    if (varType != "Real")
    {
        return index + 2;
    }

    if (isInputVariable(varName))
    {
        int varIndex = extractVariableIndex(varName);

        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }
    }

    return index + 2;
}

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

int VnnLibInputParser::parseCondition(int index,
                                     const Vector<String> &tokens,
                                     Vector<InputBoundInfo> &inputBounds)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    if (op == "and" || op == "AND" || op == "And")
    {
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
        // OR branches may embed input bounds alongside output constraints
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
        ++index;
        if ((unsigned)index + 1 >= tokens.size())
            return index;

        String token1 = tokens[index];
        String token2 = tokens[index + 1];

        String varName;
        String valueStr;
        bool varFirst = true;

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
            index += 2;
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            return index + 1;
        }

        int varIndex = extractVariableIndex(varName);

        while ((unsigned)varIndex >= inputBounds.size())
        {
            inputBounds.append(InputBoundInfo());
        }

        double value = extractScalar(valueStr);

        if (op == "<=")
        {
            if (varFirst)
            {
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
            }
            else
            {
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
            }
        }
        else // op == ">="
        {
            if (varFirst)
            {
                inputBounds[varIndex].hasLowerBound = true;
                inputBounds[varIndex].lowerBound = value;
            }
            else
            {
                inputBounds[varIndex].hasUpperBound = true;
                inputBounds[varIndex].upperBound = value;
            }
        }

        index += 2;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

BoundedTensor<torch::Tensor> VnnLibInputParser::parseInputBounds(const String &vnnlibFilePath,
                                                                  unsigned expectedInputSize)
{
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

    Vector<InputBoundInfo> inputBounds;
    parseTokens(tokens, inputBounds);

    while (inputBounds.size() < expectedInputSize)
    {
        inputBounds.append(InputBoundInfo());
    }

    // double precision first to avoid truncation during parsing
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

    // match Python float32 precision
    lowerBounds = lowerBounds.to(torch::kFloat32);
    upperBounds = upperBounds.to(torch::kFloat32);

    return BoundedTensor<torch::Tensor>(lowerBounds, upperBounds);
}

bool VnnLibInputParser::isOutputVariable(const String &varName)
{
    List<String> parts = varName.tokenize("_");
    if (parts.size() != 2)
        return false;

    return parts.front() == "Y";
}

bool VnnLibInputParser::isScalar(const String &token)
{
    if (token.length() == 0)
        return false;

    unsigned start = 0;
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

int VnnLibInputParser::parseLinearExpression(int index,
                                             const Vector<String> &tokens,
                                             Vector<NLR::OutputTerm> &terms,
                                             double &scalarSum)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &token = tokens[index];

    if (isOutputVariable(token))
    {
        int varIndex = extractVariableIndex(token);
        terms.append(NLR::OutputTerm(varIndex, 1.0));
        return index + 1;
    }

    if (isScalar(token))
    {
        scalarSum += extractScalar(token);
        return index + 1;
    }

    if (token == "(")
    {
        ++index;
        if ((unsigned)index >= tokens.size())
            return index;

        const String &op = tokens[index];
        ++index;

        if (op == "+")
        {
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
            Vector<NLR::OutputTerm> firstTerms;
            double firstScalar = 0.0;
            index = parseLinearExpression(index, tokens, firstTerms, firstScalar);
            for (unsigned i = 0; i < firstTerms.size(); ++i)
            {
                terms.append(firstTerms[i]);
            }
            scalarSum += firstScalar;

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
                scalarSum -= secondScalar;
            }

            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            if ((unsigned)index < tokens.size() && tokens[index] == ")")
                ++index;
            return index;
        }
        else if (op == "*")
        {
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
                    Vector<NLR::OutputTerm> subTerms;
                    double subScalar = 0.0;
                    index = parseLinearExpression(index, tokens, subTerms, subScalar);
                    for (unsigned i = 0; i < subTerms.size(); ++i)
                    {
                        NLR::OutputTerm scaledTerm = subTerms[i];
                        scaledTerm.coefficient *= coefficient;
                        terms.append(scaledTerm);
                    }
                    scalarSum += subScalar * coefficient;
                    coefficient = 1.0;
                    varIndex = -1;
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

    return index + 1;
}

int VnnLibInputParser::parseOutputCondition(int index,
                                            const Vector<String> &tokens,
                                            NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    if (op == "and" || op == "AND" || op == "And")
    {
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
    else if (op == "or" || op == "OR" || op == "Or")
    {
        ++index;
        
        unsigned outputDim = outputConstraints.getOutputDimension();
        Vector<Vector<NLR::OutputConstraint>> branches;
        
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
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

        Vector<NLR::OutputTerm> lhsTerms;
        Vector<NLR::OutputTerm> rhsTerms;
        double lhsScalar = 0.0;
        double rhsScalar = 0.0;

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

        bool hasOutputVar = (lhsTerms.size() > 0 || rhsTerms.size() > 0);

        if (hasOutputVar)
        {
            NLR::OutputConstraint constraint;

            // normalize to C*y <= threshold
            for (unsigned i = 0; i < lhsTerms.size(); ++i)
            {
                constraint.terms.append(lhsTerms[i]);
            }

            for (unsigned i = 0; i < rhsTerms.size(); ++i)
            {
                NLR::OutputTerm negated = rhsTerms[i];
                negated.coefficient *= -1.0;
                constraint.terms.append(negated);
            }

            double threshold = rhsScalar - lhsScalar;

            if (op == ">=")
            {
                for (unsigned i = 0; i < constraint.terms.size(); ++i)
                {
                    constraint.terms[i].coefficient *= -1.0;
                }
                threshold = -threshold;
            }

            constraint.threshold = threshold;
            outputConstraints.addConstraint(constraint);
        }

        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

int VnnLibInputParser::parseOutputBranch(int index,
                                          const Vector<String> &tokens,
                                          Vector<NLR::OutputConstraint> &branchConstraints,
                                          unsigned outputDim)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

    if (op == "and" || op == "AND" || op == "And")
    {
        ++index;
        while ((unsigned)index < tokens.size() && tokens[index] != ")")
        {
            if (tokens[index] == "(")
            {
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
        ++index;
        if ((unsigned)index >= tokens.size())
            return index;

        Vector<NLR::OutputTerm> lhsTerms;
        Vector<NLR::OutputTerm> rhsTerms;
        double lhsScalar = 0.0;
        double rhsScalar = 0.0;

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

        bool hasOutputVar = (lhsTerms.size() > 0 || rhsTerms.size() > 0);

        if (hasOutputVar)
        {
            NLR::OutputConstraint constraint;

            // normalize to C*y <= threshold
            for (unsigned i = 0; i < lhsTerms.size(); ++i)
            {
                constraint.terms.append(lhsTerms[i]);
            }

            for (unsigned i = 0; i < rhsTerms.size(); ++i)
            {
                NLR::OutputTerm negated = rhsTerms[i];
                negated.coefficient *= -1.0;
                constraint.terms.append(negated);
            }

            double threshold = rhsScalar - lhsScalar;

            if (op == ">=")
            {
                for (unsigned i = 0; i < constraint.terms.size(); ++i)
                {
                    constraint.terms[i].coefficient *= -1.0;
                }
                threshold = -threshold;
            }

            constraint.threshold = threshold;
            branchConstraints.append(constraint);
        }

        while ((unsigned)index < tokens.size() && tokens[index] != ")")
            ++index;
        return index + 1;
    }

    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

NLR::OutputConstraintSet VnnLibInputParser::parseOutputConstraints(const String &vnnlibFilePath,
                                                                    unsigned expectedOutputSize)
{
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

    NLR::OutputConstraintSet outputConstraints;
    outputConstraints.setOutputDimension(expectedOutputSize);
    parseOutputTokens(tokens, outputConstraints);

    return outputConstraints;
}

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

int VnnLibInputParser::parseConditionBoth(int index,
                                          const Vector<String> &tokens,
                                          Vector<InputBoundInfo> &inputBounds,
                                          NLR::OutputConstraintSet &outputConstraints)
{
    if ((unsigned)index >= tokens.size())
        return index;

    const String &op = tokens[index];

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
    else if (op == "or" || op == "OR" || op == "Or")
    {
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
                        ++index;
                        Vector<NLR::OutputConstraint> branchConstraints;

                        while ((unsigned)index < tokens.size() && tokens[index] != ")")
                        {
                            if (tokens[index] == "(")
                            {
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
                                    index = parseCondition(index + 1, tokens, inputBounds);
                                }
                                else if (subHasOutput)
                                {
                                    index = parseOutputBranch(index + 1, tokens, branchConstraints, outputDim);
                                }
                                else
                                {
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

            for (unsigned i = 0; i < branches.size(); ++i)
            {
                outputConstraints.addORBranch(branches[i]);
            }

            return index + 1;
        }
        else
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
    }
    else if (op == "<=" || op == ">=")
    {
        int peekIndex = index + 1;
        bool hasInputVar = false;
        bool hasOutputVar = false;

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

        if (hasInputVar && !hasOutputVar)
        {
            return parseCondition(index, tokens, inputBounds);
        }
        else if (hasOutputVar && !hasInputVar)
        {
            return parseOutputCondition(index, tokens, outputConstraints);
        }
        else
        {
            ++index;
            while ((unsigned)index < tokens.size() && tokens[index] != ")")
                ++index;
            return index + 1;
        }
    }

    while ((unsigned)index < tokens.size() && tokens[index] != ")")
        ++index;
    return index + 1;
}

std::pair<BoundedTensor<torch::Tensor>, NLR::OutputConstraintSet>
VnnLibInputParser::parseInputAndOutputConstraints(const String &vnnlibFilePath,
                                                  unsigned expectedInputSize,
                                                  unsigned expectedOutputSize)
{
    String content = readVnnlibFile(vnnlibFilePath);
    Vector<String> tokens = tokenize(content);

    Vector<InputBoundInfo> inputBounds;
    NLR::OutputConstraintSet outputConstraints;
    outputConstraints.setOutputDimension(expectedOutputSize);

    parseTokensBoth(tokens, inputBounds, outputConstraints);

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
