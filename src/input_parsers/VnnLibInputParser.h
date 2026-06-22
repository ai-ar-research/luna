/*********************                                                        */
/*! \file VnnLibInputParser.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __VnnLibInputParser_h__
#define __VnnLibInputParser_h__

#include "MString.h"
#include "Vector.h"
#include "BoundedTensor.h"
#include "OutputConstraint.h"

// avoid macro conflict with PyTorch
#ifdef Warning
#undef Warning
#endif
#ifdef LOG
#undef LOG
#endif

#include <torch/torch.h>
#include <utility>

class VnnLibInputParser
{
public:
    static BoundedTensor<torch::Tensor> parseInputBounds(const String &vnnlibFilePath,
                                                          unsigned expectedInputSize);

    static NLR::OutputConstraintSet parseOutputConstraints(const String &vnnlibFilePath,
                                                           unsigned expectedOutputSize);

    static std::pair<BoundedTensor<torch::Tensor>, NLR::OutputConstraintSet>
    parseInputAndOutputConstraints(const String &vnnlibFilePath,
                                   unsigned expectedInputSize,
                                   unsigned expectedOutputSize);

private:
    struct InputBoundInfo {
        bool hasLowerBound;
        bool hasUpperBound;
        double lowerBound;
        double upperBound;

        InputBoundInfo()
            : hasLowerBound(false), hasUpperBound(false),
              lowerBound(-std::numeric_limits<double>::infinity()),
              upperBound(std::numeric_limits<double>::infinity()) {}
    };

    static String readVnnlibFile(const String &vnnlibFilePath);
    static Vector<String> tokenize(const String &vnnlibContent);

    static void parseTokens(const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    static int parseCommand(int index,
                           const Vector<String> &tokens,
                           Vector<InputBoundInfo> &inputBounds);

    static int parseDeclareConst(int index,
                                 const Vector<String> &tokens,
                                 Vector<InputBoundInfo> &inputBounds);

    static int parseAssert(int index,
                          const Vector<String> &tokens,
                          Vector<InputBoundInfo> &inputBounds);

    static int parseCondition(int index,
                             const Vector<String> &tokens,
                             Vector<InputBoundInfo> &inputBounds);

    static int extractVariableIndex(const String &varName);
    static bool isInputVariable(const String &varName);
    static bool isOutputVariable(const String &varName);
    static double extractScalar(const String &token);

    static void parseOutputTokens(const Vector<String> &tokens,
                                  NLR::OutputConstraintSet &outputConstraints);

    static int parseOutputCommand(int index,
                                  const Vector<String> &tokens,
                                  NLR::OutputConstraintSet &outputConstraints);

    static int parseOutputAssert(int index,
                                 const Vector<String> &tokens,
                                 NLR::OutputConstraintSet &outputConstraints);

    static int parseOutputCondition(int index,
                                    const Vector<String> &tokens,
                                    NLR::OutputConstraintSet &outputConstraints);

    static int parseOutputBranch(int index,
                                  const Vector<String> &tokens,
                                  Vector<NLR::OutputConstraint> &branchConstraints,
                                  unsigned outputDim);

    static int parseLinearExpression(int index,
                                     const Vector<String> &tokens,
                                     Vector<NLR::OutputTerm> &terms,
                                     double &scalarSum);

    static bool isScalar(const String &token);

    static void parseTokensBoth(const Vector<String> &tokens,
                                Vector<InputBoundInfo> &inputBounds,
                                NLR::OutputConstraintSet &outputConstraints);

    static int parseCommandBoth(int index,
                                const Vector<String> &tokens,
                                Vector<InputBoundInfo> &inputBounds,
                                NLR::OutputConstraintSet &outputConstraints);

    static int parseAssertBoth(int index,
                               const Vector<String> &tokens,
                               Vector<InputBoundInfo> &inputBounds,
                               NLR::OutputConstraintSet &outputConstraints);

    static int parseConditionBoth(int index,
                                  const Vector<String> &tokens,
                                  Vector<InputBoundInfo> &inputBounds,
                                  NLR::OutputConstraintSet &outputConstraints);
};

#endif // __VnnLibInputParser_h__
