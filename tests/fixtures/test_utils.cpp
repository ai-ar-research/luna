#include "test_utils.h"
#include "src/input_parsers/VnnLibInputParser.h"
#include "src/engine/CROWNAnalysis.h"

using VnnLibInputParser = ::VnnLibInputParser;
#include <torch/torch.h>
#include <random>
#include <algorithm>

namespace test
{

BoundedTensor<torch::Tensor> BoundGenerator::epsilonBall( const torch::Tensor &center,
                                                          double epsilon )
{
    torch::Tensor lower = center - epsilon;
    torch::Tensor upper = center + epsilon;
    return BoundedTensor<torch::Tensor>( lower, upper );
}

BoundedTensor<torch::Tensor> BoundGenerator::randomBounds( const std::vector<int64_t> &shape,
                                                           double minVal, double maxVal,
                                                           double width )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> centerDist( minVal, maxVal );

    torch::Tensor lower = torch::zeros( shape, torch::kFloat32 );
    torch::Tensor upper = torch::zeros( shape, torch::kFloat32 );

    auto lowerFlat = lower.view( { -1 } );
    auto upperFlat = upper.view( { -1 } );
    auto lowerAccessor = lowerFlat.accessor<float, 1>();
    auto upperAccessor = upperFlat.accessor<float, 1>();

    for ( int64_t i = 0; i < lowerFlat.numel(); ++i )
    {
        double center = centerDist( gen );
        lowerAccessor[i] = center - width / 2.0;
        upperAccessor[i] = center + width / 2.0;
    }

    return BoundedTensor<torch::Tensor>( lower, upper );
}

BoundedTensor<torch::Tensor> BoundGenerator::wideBounds( const std::vector<int64_t> &shape,
                                                         double center, double halfWidth )
{
    torch::Tensor lower = torch::full( shape, center - halfWidth, torch::kFloat32 );
    torch::Tensor upper = torch::full( shape, center + halfWidth, torch::kFloat32 );
    return BoundedTensor<torch::Tensor>( lower, upper );
}

BoundedTensor<torch::Tensor> BoundGenerator::fromVNNLib( const std::string &vnnlibPath )
{
    // Note: This is a simplified version - in practice, you'd need to know the input size
    // For full implementation, you'd parse the ONNX file first or pass input size as parameter
    // For now, we'll try to parse with a reasonable default and handle errors
    // In actual usage, this should be used with models that have known input sizes
    unsigned defaultInputSize = 784; // Common MNIST size, but may vary
    try
    {
        return VnnLibInputParser::parseInputBounds( String( vnnlibPath.c_str() ),
                                                    defaultInputSize );
    }
    catch ( ... )
    {
        // If default fails, user should provide input size - for now throw
        throw std::runtime_error( "fromVNNLib requires known input size - use parseInputBounds "
                                  "directly with model input size" );
    }
}

bool SoundnessChecker::boundsContainValue( const BoundedTensor<torch::Tensor> &bounds,
                                           const torch::Tensor &value )
{
    torch::Tensor lower = bounds.lower().flatten();
    torch::Tensor upper = bounds.upper().flatten();
    torch::Tensor valFlat = value.flatten();

    int64_t n = std::min( { lower.numel(), upper.numel(), valFlat.numel() } );

    for ( int64_t i = 0; i < n; ++i )
    {
        float vi = valFlat[i].item<float>();
        float li = lower[i].item<float>();
        float ui = upper[i].item<float>();
        if ( vi < li || vi > ui )
        {
            return false;
        }
    }
    return true;
}

bool SoundnessChecker::verifySoundnessBySampling( NLR::TorchModel &model,
                                                  const BoundedTensor<torch::Tensor> &inputBounds,
                                                  const BoundedTensor<torch::Tensor> &outputBounds,
                                                  unsigned numSamples, unsigned seed )
{

    std::mt19937 gen( seed );
    std::uniform_real_distribution<double> dis( 0.0, 1.0 );

    torch::Tensor lower = inputBounds.lower().flatten();
    torch::Tensor upper = inputBounds.upper().flatten();

    // Note: Thread configuration should be set by test fixtures in SetUp()
    // to avoid "cannot set number of interop threads after parallel work has started" errors

    for ( unsigned s = 0; s < numSamples; ++s )
    {
        // Sample uniformly from input bounds
        torch::Tensor sample = torch::zeros_like( lower );
        for ( int64_t i = 0; i < sample.numel(); ++i )
        {
            double alpha = dis( gen );
            double val = lower[i].item<float>() * ( 1.0 - alpha ) + upper[i].item<float>() * alpha;
            sample[i] = val;
        }

        // Reshape if needed (for image inputs)
        torch::Tensor sampleReshaped = sample;
        if ( sample.dim() == 1 && sample.numel() == 3072 )
        {
            // Common CIFAR shape: 3*32*32 = 3072
            sampleReshaped = sample.view( { 1, 3, 32, 32 } );
        }
        else if ( sample.dim() == 1 )
        {
            sampleReshaped = sample.unsqueeze( 0 );
        }

        // Forward pass
        torch::NoGradGuard no_grad;
        Map<unsigned, torch::Tensor> activations =
            model.forwardAndStoreActivations( sampleReshaped );
        torch::Tensor output = activations[model.getOutputIndex()];
        output = output.flatten().to( torch::kFloat32 );

        // Check containment
        if ( !boundsContainValue( outputBounds, output ) )
        {
            return false;
        }
    }

    return true;
}

bool SoundnessChecker::centerContainedInBounds( const BoundedTensor<torch::Tensor> &inputBounds,
                                                const BoundedTensor<torch::Tensor> &outputBounds,
                                                NLR::TorchModel &model )
{

    torch::NoGradGuard no_grad;
    torch::Tensor center = inputBounds.center();

    // Reshape if needed
    torch::Tensor centerReshaped = center;
    if ( center.dim() == 1 && center.numel() == 3072 )
    {
        centerReshaped = center.view( { 1, 3, 32, 32 } );
    }
    else if ( center.dim() == 1 )
    {
        centerReshaped = center.unsqueeze( 0 );
    }

    Map<unsigned, torch::Tensor> activations = model.forwardAndStoreActivations( centerReshaped );
    torch::Tensor output = activations[model.getOutputIndex()];
    output = output.flatten().to( torch::kFloat32 );

    return boundsContainValue( outputBounds, output );
}

bool SoundnessChecker::crownTighterThanIBP( const BoundedTensor<torch::Tensor> &ibpBounds,
                                            const BoundedTensor<torch::Tensor> &crownBounds )
{

    torch::Tensor ibpLower = ibpBounds.lower().flatten();
    torch::Tensor ibpUpper = ibpBounds.upper().flatten();
    torch::Tensor crownLower = crownBounds.lower().flatten();
    torch::Tensor crownUpper = crownBounds.upper().flatten();

    int64_t n =
        std::min( { ibpLower.numel(), ibpUpper.numel(), crownLower.numel(), crownUpper.numel() } );

    for ( int64_t i = 0; i < n; ++i )
    {
        float ibpL = ibpLower[i].item<float>();
        float ibpU = ibpUpper[i].item<float>();
        float crownL = crownLower[i].item<float>();
        float crownU = crownUpper[i].item<float>();

        // CROWN should be at least as tight: crownL >= ibpL and crownU <= ibpU
        if ( crownL < ibpL || crownU > ibpU )
        {
            return false;
        }
    }

    return true;
}

bool SoundnessChecker::alphaCrownTighterThanCrown(
    const BoundedTensor<torch::Tensor> &crownBounds,
    const BoundedTensor<torch::Tensor> &alphaCrownBounds )
{

    torch::Tensor crownLower = crownBounds.lower().flatten();
    torch::Tensor crownUpper = crownBounds.upper().flatten();
    torch::Tensor alphaLower = alphaCrownBounds.lower().flatten();
    torch::Tensor alphaUpper = alphaCrownBounds.upper().flatten();

    int64_t n = std::min(
        { crownLower.numel(), crownUpper.numel(), alphaLower.numel(), alphaUpper.numel() } );

    for ( int64_t i = 0; i < n; ++i )
    {
        float crownL = crownLower[i].item<float>();
        float crownU = crownUpper[i].item<float>();
        float alphaL = alphaLower[i].item<float>();
        float alphaU = alphaUpper[i].item<float>();

        // AlphaCROWN should be at least as tight: alphaL >= crownL and alphaU <= crownU
        if ( alphaL < crownL || alphaU > crownU )
        {
            return false;
        }
    }

    return true;
}

} // namespace test
