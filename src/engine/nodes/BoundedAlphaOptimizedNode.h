#ifndef __BoundedAlphaOptimizeNode_h__
#define __BoundedAlphaOptimizeNode_h__

#include "BoundedTorchNode.h"
#include "MStringf.h"
#include "LunaConfiguration.h"

// Undefine Warning macro to avoid conflict with PyTorch
#ifdef Warning
#undef Warning
#endif

#include <torch/torch.h>
#include <string>

// Undefine LOG macro from PyTorch before including Debug.h
#ifdef LOG
#undef LOG
#endif

#include "Debug.h"

// Redefine Warning macro for CVC4 compatibility
#ifndef Warning
#define Warning (! ::CVC4::WarningChannel.isOn()) ? ::CVC4::nullCvc4Stream : ::CVC4::WarningChannel
#endif

namespace NLR {

// Forward declarations
class AlphaCROWNAnalysis;

class BoundedAlphaOptimizeNode : public BoundedTorchNode
{
public:
    BoundedAlphaOptimizeNode() : _alphaCrownAnalysis(nullptr), _optimizationStage("") {}

    virtual ~BoundedAlphaOptimizeNode() {}
    
    // Alpha-CROWN integration
    void setAlphaCrownAnalysis(AlphaCROWNAnalysis* analysis) { _alphaCrownAnalysis = analysis; }
    AlphaCROWNAnalysis* getAlphaCrownAnalysis() const { return _alphaCrownAnalysis; }
    bool isAlphaOptimizationEnabled() const { return _alphaCrownAnalysis != nullptr; }

    // Optimization side queries (delegated to AlphaCROWNAnalysis)
    bool isOptimizingLower() const;
    bool isOptimizingUpper() const;
    bool isOptimizingBoth() const;
    
    // Optimization stage management
    std::string getOptimizationStage() const { return _optimizationStage; }
    void setOptimizationStage(const std::string& stage) { _optimizationStage = stage; }
    
    // Alpha initialization support (following auto_LiRPA)
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
    
    // CROWN slope access for alpha initialization (following auto_LiRPA approach)
    virtual torch::Tensor getCROWNSlope(bool isLowerBound) const = 0;

protected:
    // Alpha-CROWN support
    AlphaCROWNAnalysis* _alphaCrownAnalysis;
    
    // Optimization stage: "init", "opt", "reuse", or ""
    std::string _optimizationStage;
    
    // CROWN slopes saved during initialization (following auto_LiRPA)
    // This becomes the alpha initialization values (auto_LiRPA: self.init_d = lower_d)
    torch::Tensor init_d;

    void storeInitD(const torch::Tensor& crown_slopes) {
        // Store CROWN slopes during initialization stage (following auto_LiRPA)
        // This is called when _optimizationStage == "init" 
        // auto_LiRPA: self.init_d = lower_d (becomes alpha initialization values)
        
        if (crown_slopes.defined() && crown_slopes.numel() > 0) {
            init_d = crown_slopes.detach().clone();
        } else {
            log(Stringf("storeInitD() - Warning: Invalid CROWN slopes for node %u", getNodeIndex()));
        }
    }

private:
    // Utility methods
    void log(const String& message) {
        (void)message;
    }
};

} // namespace NLR

#endif // __BoundedAlphaOptimizeNode_h__