/*********************                                                        */
/*! \file BoundedAlphaOptimizedNode.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __BoundedAlphaOptimizeNode_h__
#define __BoundedAlphaOptimizeNode_h__

#include "BoundedTorchNode.h"
#include "MStringf.h"
#include "LunaConfiguration.h"

#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <string>

#ifdef LOG
#undef LOG
#endif

#include "Debug.h"

#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR {

class AlphaCROWNAnalysis;

class BoundedAlphaOptimizeNode : public BoundedTorchNode
{
public:
    BoundedAlphaOptimizeNode() : _alphaCrownAnalysis(nullptr), _optimizationStage("") {}

    virtual ~BoundedAlphaOptimizeNode() {}

    void setAlphaCrownAnalysis(AlphaCROWNAnalysis* analysis) { _alphaCrownAnalysis = analysis; }
    AlphaCROWNAnalysis* getAlphaCrownAnalysis() const { return _alphaCrownAnalysis; }
    bool isAlphaOptimizationEnabled() const { return _alphaCrownAnalysis != nullptr; }

    bool isOptimizingLower() const;
    bool isOptimizingUpper() const;
    bool isOptimizingBoth() const;

    std::string getOptimizationStage() const { return _optimizationStage; }
    void setOptimizationStage(const std::string& stage) { _optimizationStage = stage; }

    torch::Tensor getInitD() const { return init_d; }
    bool hasInitD() const { return init_d.defined() && init_d.numel() > 0; }

    virtual void computeAlphaRelaxation(
        const torch::Tensor& last_lA,
        const torch::Tensor& last_uA,
        const torch::Tensor& input_lower,
        const torch::Tensor& input_upper,
        torch::Tensor& d_lower,
        torch::Tensor& d_upper,
        torch::Tensor& bias_lower,
        torch::Tensor& bias_upper) = 0;

    virtual torch::Tensor getCROWNSlope(bool isLowerBound) const = 0;

protected:
    AlphaCROWNAnalysis* _alphaCrownAnalysis;

    // "init", "opt", "reuse", or ""
    std::string _optimizationStage;

    // CROWN slopes from init stage become alpha initialization values
    torch::Tensor init_d;

    void storeInitD(const torch::Tensor& crown_slopes) {
        if (crown_slopes.defined() && crown_slopes.numel() > 0) {
            init_d = crown_slopes.detach().clone();
        } else {
            log(Stringf("storeInitD() - Warning: Invalid CROWN slopes for node %u", getNodeIndex()));
        }
    }

private:
    void log(const String& message) {
        (void)message;
    }
};

} // namespace NLR

#endif // __BoundedAlphaOptimizeNode_h__
