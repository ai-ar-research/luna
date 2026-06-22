/*********************                                                        */
/*! \file AlphaCROWNAnalysis.cpp
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#include "AlphaCROWNAnalysis.h"
#include "nodes/BoundedAlphaOptimizedNode.h"
#include "nodes/BoundedSigmoidNode.h"

#include "Debug.h"
#include "MStringf.h"
#include "LunaError.h"
#include "TimeUtils.h"

#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace NLR {

AlphaCROWNAnalysis::AlphaCROWNAnalysis(TorchModel* torchModel)
    : _torchModel(torchModel)
    , _alphaEnabled(true)
    , _initialized(false)
    , _iteration(LunaConfiguration::ALPHA_ITERATIONS)
    , _learningRate(LunaConfiguration::ALPHA_LR)
    , _optimizationStage("init")
{
    if (!_torchModel) {
        throw LunaError(LunaError::UNINITIALIZED_NODE, "AlphaCROWNAnalysis requires a valid TorchModel instance");
    }

    _crownAnalysis = std::make_unique<CROWNAnalysis>(_torchModel);
    updateFromConfig();
}

AlphaCROWNAnalysis::~AlphaCROWNAnalysis()
{
}

void AlphaCROWNAnalysis::initializeAlphaParameters()
{
    log("initializeAlphaParameters() - Starting alpha parameter initialization");

    if (_initialized) {
        log("initializeAlphaParameters() - Alpha parameters already initialized");
        return;
    }

    try {
        log("initializeAlphaParameters() - Forward pass");
        performForwardPass();

        log("initializeAlphaParameters() - Prepare optimizable activations");
        prepareOptimizableActivations();

        log("initializeAlphaParameters() - Standard CROWN initialization pass");
        performCROWNInitializationPass();

        log("initializeAlphaParameters() - Alpha parameters will be created lazily per start");

        _initialized = true;
        log("initializeAlphaParameters() - Alpha parameter initialization completed successfully");
    } catch (const std::exception& e) {
        log(Stringf("initializeAlphaParameters() - Exception: %s", e.what()));
        throw;
    }
}

void AlphaCROWNAnalysis::performForwardPass()
{
    log("performForwardPass() - forward pass through model");

    // autograd required for gradient flow through alpha parameters during optimization
    torch::GradMode::set_enabled(true);

    log("performForwardPass() - Forward pass structure established with autograd enabled");
}

void AlphaCROWNAnalysis::prepareOptimizableActivations()
{
    log("prepareOptimizableActivations() - Finding alpha-optimizable activation nodes");

    _optimizableNodes.clear();

    unsigned numNodes = _crownAnalysis->getNumNodes();
    for ( unsigned i = 0; i < numNodes; ++i )
    {
        auto node = _crownAnalysis->getNode(i);
        if ( !node )
        {
            continue;
        }

        auto alphaNode = std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>(node);
        if ( alphaNode )
        {
            _optimizableNodes.push_back(std::make_pair(i, alphaNode));

            alphaNode->setOptimizationStage("init");
            alphaNode->setAlphaCrownAnalysis(this);

            log(Stringf("prepareOptimizableActivations() - Added optimizable node %u (type=%d)", i, (int)node->getNodeType()));
        }
    }

    log(Stringf("prepareOptimizableActivations() - Found %u optimizable nodes", (unsigned)_optimizableNodes.size()));
}

void AlphaCROWNAnalysis::performCROWNInitializationPass()
{
    log("performCROWNInitializationPass() - Running standard CROWN to capture relaxation slopes");

    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;

    _crownAnalysis->resetProcessingState();

    _crownAnalysis->run(false);

    log("performCROWNInitializationPass() - CROWN initialization pass completed");
}


AlphaParameters& AlphaCROWNAnalysis::ensureAlphaFor(
    unsigned nodeIndex,
    const std::string& startKey,
    int specDim, int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{
    auto& perStart = _alphaByNodeStart[nodeIndex];
    auto it = perStart.find(startKey);

    auto input_lb_flat = input_lb.detach().flatten();
    auto input_ub_flat = input_ub.detach().flatten();
    torch::Tensor unstableMask;
    auto nodePtr = _crownAnalysis->getNode(nodeIndex);
    if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
        unstableMask = (input_ub_flat - input_lb_flat) > 1e-6;
    } else {
        unstableMask = (input_lb_flat < 0) & (input_ub_flat > 0);
    }
    int numUnstable = unstableMask.sum().item<int>();
    bool need_new = (it == perStart.end());
    if (!need_new) {
        const auto& ap = it->second;
        bool spec_mismatch = false;
        if (ap.alpha.defined()) {
            const auto storedSpecDim = (int)ap.alpha.size(1);

            spec_mismatch = (storedSpecDim != specDim)
                && !(ap.hasSpecDefaultSlot && storedSpecDim == specDim + 1);
        }
        if (spec_mismatch) {
            perStart.erase(it);
            need_new = true;
        } else {
            return it->second;
        }
    }

    if (need_new) {
        auto options = input_lb.options().dtype(torch::kFloat32).requires_grad(false);

        AlphaParameters params;
        params.specDim = specDim;
        params.outDim = outDim;
        params.numUnstable = numUnstable;
        params.unstableMask = unstableMask.detach().clone();

        if (numUnstable == 0) {
            if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
                params.alpha = torch::empty({2, specDim, 1, 0, 8}, options);
            } else {
                params.alpha = torch::empty({2, specDim, 1, 0}, options);
            }
            params.unstableIndices = torch::empty({0}, torch::kLong);
            perStart[startKey] = std::move(params);
            return perStart[startKey];
        }

        params.unstableIndices = torch::nonzero(unstableMask).flatten().to(torch::kLong).detach();

        torch::Tensor slope_init;
        torch::Tensor full_slope_debug;
        {
            std::shared_ptr<BoundedAlphaOptimizeNode> nodePtr;
            for (auto &p : _optimizableNodes) {
                if (p.first == nodeIndex) {
                    nodePtr = p.second;
                    break;
                }
            }
            if (nodePtr) {
                torch::Tensor full_slope = nodePtr->getCROWNSlope(true);
                full_slope_debug = full_slope;
                if (full_slope.defined() && full_slope.numel() > 0) {
                    slope_init = full_slope.flatten().index_select(0, params.unstableIndices);
                } else {
                    slope_init = torch::full({numUnstable}, 0.5f, options);
                }
            } else {
                slope_init = torch::full({numUnstable}, 0.5f, options);
            }
        }

        bool useSparseSpec = false;
        Vector<unsigned> cachedUnstableSpecs;
        bool cachedSparseMode = false;
        unsigned cachedNodeSize = 0;
        if (_crownAnalysis->getAlphaStartCacheInfo(startKey, cachedUnstableSpecs, cachedSparseMode, cachedNodeSize)) {
            useSparseSpec = cachedSparseMode && (int)cachedUnstableSpecs.size() == specDim;
        }

        // sigmoid: 8 slots per neuron (auto_LiRPA s-shaped tangent-point style)
        int alphaSlots = 1;
        if (nodePtr && nodePtr->getNodeType() == NodeType::SIGMOID) {
            alphaSlots = 8;
        }

        int specDimAlpha = useSparseSpec ? (specDim + 1) : specDim;
        torch::Tensor alpha;
        torch::Tensor expanded;
        if (alphaSlots == 1) {
            alpha = torch::zeros({2, specDimAlpha, 1, numUnstable}, options);
            expanded = slope_init.view({1, 1, 1, numUnstable})
                        .expand({2, specDim, 1, numUnstable})
                        .contiguous()
                        .clone()
                        .to(options.dtype());

        } else {
            auto mid_unstable = ((input_lb_flat + input_ub_flat) * 0.5f)
                .index_select(0, params.unstableIndices)
                .to(options.dtype());
            torch::Tensor d_lower_unstable = mid_unstable.clone();
            torch::Tensor d_upper_unstable = mid_unstable.clone();
            if (nodePtr) {
                auto sigmoidNode = std::dynamic_pointer_cast<BoundedSigmoidNode>(nodePtr);
                if (sigmoidNode) {
                    auto d_pair = sigmoidNode->getDefaultTangentPoints(input_lb, input_ub);
                    d_lower_unstable = d_pair.first.flatten().index_select(0, params.unstableIndices).to(options.dtype());
                    d_upper_unstable = d_pair.second.flatten().index_select(0, params.unstableIndices).to(options.dtype());
                }
            }

            alpha = torch::zeros({2, specDimAlpha, 1, numUnstable, alphaSlots}, options);
            auto expanded_mid = mid_unstable.view({1, 1, 1, numUnstable})
                .expand({2, specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());
            auto expanded_d_lower = d_lower_unstable.view({1, 1, 1, numUnstable})
                .expand({2, specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());
            auto expanded_d_upper = d_upper_unstable.view({1, 1, 1, numUnstable})
                .expand({2, specDim, 1, numUnstable})
                .contiguous()
                .clone()
                .to(options.dtype());

            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 1}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 2}, expanded_mid);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 3}, expanded_mid);
            // auto_LiRPA style: tp_both_lower/tp_both_upper from precomputed defaults
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 4}, expanded_d_lower);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 5}, expanded_d_lower);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 6}, expanded_d_upper);
            alpha.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 7}, expanded_d_upper);
            expanded = alpha.narrow(1, 0, specDim);
        }
        if (useSparseSpec) {
            alpha.narrow(1, 1, specDim).copy_(expanded);
            params.hasSpecDefaultSlot = true;
        } else {
            alpha.copy_(expanded);
        }

        // detach() makes alpha a proper leaf tensor; without it, clamp creates a
        // non-leaf that triggers "modified by inplace operation" errors from optimizer
        if (alphaSlots == 1) {
            alpha = torch::clamp(alpha, 0.0f, 1.0f).detach();
        } else {
            alpha = alpha.detach();
        }
        alpha.set_requires_grad(true);

        params.alpha = alpha;
        perStart[startKey] = std::move(params);

    }
    return perStart[startKey];
}

torch::Tensor AlphaCROWNAnalysis::computeOptimizedBounds(LunaConfiguration::BoundSide side)
{
    log(Stringf("computeOptimizedBounds() - Starting optimization for %s bounds",
                side == LunaConfiguration::BoundSide::Lower ? "LOWER" : "UPPER"));
    if (!_initialized) {
        log("computeOptimizedBounds() - Initializing alpha parameters");
        initializeAlphaParameters();
    }

    if (!shouldPerformOptimization()) {
        log("computeOptimizedBounds() - Alpha optimization not applicable, falling back to standard CROWN");
        _crownAnalysis->run(false);
        return side == LunaConfiguration::BoundSide::Lower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    LunaConfiguration::BOUND_SIDE = side;
    bool isLower = (side == LunaConfiguration::BoundSide::Lower);

    auto seededBounds = _torchModel->hasPersistedIntermediateBounds()
        ? _torchModel->getPersistedIntermediateBounds()
        : std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>{};

    _bestIntermediateBounds.clear();
    _crownAnalysis->clearAllNodeBounds();
    setOptimizationStage("opt");
    LunaConfiguration::ENABLE_FIRST_LINEAR_IBP = false;

    _crownAnalysis->setAlphaStartCacheEnabled(true);
    _crownAnalysis->clearConcreteBounds();
    _crownAnalysis->clearCrownState();
    _crownAnalysis->resetProcessingState();

    if (!seededBounds.empty() && !LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS) {
        for (auto& [nodeIdx, pair] : seededBounds) {
            _bestIntermediateBounds[nodeIdx] = {pair.first.detach().clone(), pair.second.detach().clone()};
        }
        restoreBestIntermediateBounds();
    }

    _crownAnalysis->run(true);

    snapshotBestIntermediateBounds();

    // STE stabilization: capture initial bounds, tighten monotonically
    if (LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS) {
        _referenceBounds.clear();
        for (const auto& [nodeIdx, bounds] : _bestIntermediateBounds) {
            _referenceBounds[nodeIdx] = {bounds.first.detach().clone(), bounds.second.detach().clone()};
        }
        _crownAnalysis->setReferenceBounds(_referenceBounds);
    }

    torch::Tensor initialBound = isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();

    auto alphaParams = collectAlphaParameters();
    if (alphaParams.empty()) {
        _crownAnalysis->run(false);
        return isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    }

    auto optimizer = std::make_shared<torch::optim::Adam>(
        alphaParams,
        torch::optim::AdamOptions(_learningRate).betas(std::make_tuple(0.9, 0.999)).eps(1e-8));

    torch::Tensor bestBounds;
    float currentLR = _learningRate;
    const unsigned early_stop_patience = LunaConfiguration::EARLY_STOP_PATIENCE;
    unsigned iterations_without_improvement = 0;
    torch::Tensor best_loss_tensor;

    for (unsigned iter = 0; iter < _iteration; ++iter) {
        std::unique_ptr<torch::NoGradGuard> no_grad;
        if (iter == _iteration - 1) {
            no_grad = std::make_unique<torch::NoGradGuard>();
        }

        _crownAnalysis->clearConcreteBounds();
        _crownAnalysis->clearAllNodeBounds();
        _crownAnalysis->clearCrownState();
        _crownAnalysis->resetProcessingState();

        if (!LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS
            && !_bestIntermediateBounds.empty())
            restoreBestIntermediateBounds();

        _crownAnalysis->run(true);

        torch::Tensor currentBound = isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
        torch::Tensor currentBoundDetached = currentBound.detach();

        if (LunaConfiguration::STOP_ALPHA_ON_VERIFIED) {
            torch::Tensor iterLb = extractLowerBoundsFromCROWN();
            torch::Tensor iterUb = extractUpperBoundsFromCROWN();
            if (_torchModel->isSpecVerified(iterLb, iterUb)) {
                std::cout << "Verified during alpha-CROWN at iteration " << iter << std::endl;
                log(Stringf("All specs verified at iteration %u", iter));
                bestBounds = isLower
                    ? torch::max(bestBounds.defined() ? bestBounds : currentBoundDetached, currentBoundDetached)
                    : torch::min(bestBounds.defined() ? bestBounds : currentBoundDetached, currentBoundDetached);
                break;
            }
        }

        torch::Tensor loss;
        if (iter < _iteration - 1) {
            loss = isLower ? computeLoss(currentBound, torch::Tensor(), isLower)
                           : computeLoss(torch::Tensor(), currentBound, isLower);
        }

        bool improved = false;
        std::vector<int> improvedIndices;

        if (!bestBounds.defined()) {
            bestBounds = currentBoundDetached.clone();
            improved = true;
            int64_t specDim = currentBoundDetached.dim() == 0
                ? 1
                : (currentBoundDetached.dim() == 1 ? currentBoundDetached.size(0)
                                                   : currentBoundDetached.size(currentBoundDetached.dim() - 1));
            improvedIndices.reserve(static_cast<size_t>(specDim));
            for (int64_t i = 0; i < specDim; ++i) {
                improvedIndices.push_back(static_cast<int>(i));
            }
        } else {
            improvedIndices = findImprovedIndices(currentBoundDetached, bestBounds, isLower);
            improved = !improvedIndices.empty();
            if (improved) {
                bestBounds = isLower
                    ? torch::max(bestBounds, currentBoundDetached).detach()
                    : torch::min(bestBounds, currentBoundDetached).detach();
            }
        }

        bool shouldSaveBest = (iter < 1)
            || (iter > static_cast<unsigned>(_iteration * LunaConfiguration::START_SAVE_BEST))
            || (iterations_without_improvement >= early_stop_patience);

        if (improved && shouldSaveBest) {
            updateBestAlphas(improvedIndices);
            snapshotBestIntermediateBounds();

            if (LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS) {
                for (const auto& [nodeIdx, bounds] : _bestIntermediateBounds) {
                    _referenceBounds[nodeIdx] = {bounds.first.detach().clone(), bounds.second.detach().clone()};
                }
                _crownAnalysis->setReferenceBounds(_referenceBounds);
            }
        }

        if (improved) {
            if (loss.defined()) best_loss_tensor = loss.detach().abs();
            iterations_without_improvement = 0;
        } else if (loss.defined()) {
            auto current_abs_loss = loss.detach().abs();
            bool loss_improved = !best_loss_tensor.defined()
                || (current_abs_loss < best_loss_tensor).item<bool>();
            if (loss_improved) {
                best_loss_tensor = current_abs_loss;
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement++;
            }
        }

        if (iterations_without_improvement >= early_stop_patience) {
            break;
        }

        if (iter < _iteration - 1 && loss.defined()) {
            if (!loss.requires_grad()) {
                break;
            }
            optimizer->zero_grad();
            loss.backward();

            _crownAnalysis->clearConcreteBounds();
            _crownAnalysis->resetProcessingState();
            optimizer->step();
            clipAlphaParameters();

            currentLR *= LunaConfiguration::ALPHA_LR_DECAY;
            for (auto& group : optimizer->param_groups()) {
                if (auto adamGroup = dynamic_cast<torch::optim::AdamOptions*>(&group.options())) {
                    adamGroup->lr(currentLR);
                }
            }
        }
    }

    restoreBestAlphas();
    _crownAnalysis->clearReferenceBounds();
    _crownAnalysis->clearConcreteBounds();
    restoreBestIntermediateBounds();
    _crownAnalysis->resetProcessingState();
    _crownAnalysis->run(false);

    torch::Tensor finalBound = isLower ? extractLowerBoundsFromCROWN() : extractUpperBoundsFromCROWN();
    if (bestBounds.defined()) {
        finalBound = isLower ? torch::max(finalBound, bestBounds) : torch::min(finalBound, bestBounds);
    }

    log(Stringf("computeOptimizedBounds() - %s bound optimization completed",
                isLower ? "Lower" : "Upper"));

    return finalBound;
}

torch::Tensor AlphaCROWNAnalysis::extractLowerBoundsFromCROWN()
{
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if (_crownAnalysis->hasConcreteBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getConcreteLowerBound(outputIndex);

        return lowerBounds;
    }

    if (_crownAnalysis->hasIBPBounds(outputIndex)) {
        torch::Tensor lowerBounds = _crownAnalysis->getIBPLowerBound(outputIndex);
        return lowerBounds;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
    return torch::zeros({1}, options);
}

torch::Tensor AlphaCROWNAnalysis::extractUpperBoundsFromCROWN()
{
    unsigned outputIndex = _crownAnalysis->getOutputIndex();

    if (_crownAnalysis->hasConcreteBounds(outputIndex)) {
        torch::Tensor upperBounds = _crownAnalysis->getConcreteUpperBound(outputIndex);
        return upperBounds;
    }

    if (_crownAnalysis->hasIBPBounds(outputIndex)) {
        torch::Tensor upperBounds = _crownAnalysis->getIBPUpperBound(outputIndex);
        return upperBounds;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(_torchModel->getDevice());
    return torch::zeros({1}, options);
}

std::vector<torch::Tensor> AlphaCROWNAnalysis::collectAlphaParameters()
{
    std::vector<torch::Tensor> params;

    for (auto& [nodeIdx, perStart] : _alphaByNodeStart) {
        for (auto& [startKey, ap] : perStart) {
            if (ap.alpha.requires_grad()) {
                params.push_back(ap.alpha);
            }
        }
    }

    log(Stringf("collectAlphaParameters() - Collected %u alpha parameter tensors",
                (unsigned)params.size()));
    return params;
}

void AlphaCROWNAnalysis::updateBestAlphas(const std::vector<int>& improvedIndices)
{
    if (improvedIndices.empty()) {
        return;
    }

    log(Stringf("updateBestAlphas() - Updating best alphas for %zu improved specs",
                improvedIndices.size()));

    auto& currentStorage = _alphaByNodeStart;
    auto& bestStorage = _bestAlphaByNodeStart;

    torch::NoGradGuard no_grad;
    using torch::indexing::Slice;

    for (auto& [nodeIdx, perStart] : currentStorage) {
        for (auto& [startKey, ap] : perStart) {
            if (!ap.alpha.defined() || ap.alpha.numel() == 0) {
                continue;
            }

            auto& bestPerStart = bestStorage[nodeIdx];
            auto itBest = bestPerStart.find(startKey);
            if (itBest == bestPerStart.end()) {
                AlphaParameters copy = ap;
                copy.alpha = ap.alpha.detach().clone();
                bestPerStart[startKey] = std::move(copy);
                continue;
            }

            auto& bestAp = itBest->second;
            auto currentAlpha = ap.alpha.detach();

            if (!bestAp.alpha.defined() || bestAp.alpha.sizes() != currentAlpha.sizes()) {
                bestAp.alpha = currentAlpha.clone();
                continue;
            }

            auto idxTensor = torch::tensor(
                improvedIndices,
                torch::TensorOptions().dtype(torch::kLong).device(currentAlpha.device()));

            if (ap.hasSpecDefaultSlot) {
                idxTensor = idxTensor + 1;
            }

            auto validMask = idxTensor < currentAlpha.size(1);
            idxTensor = idxTensor.index({validMask});
            if (idxTensor.numel() == 0) {
                continue;
            }

            bestAp.alpha.index_put_({Slice(), idxTensor}, currentAlpha.index({Slice(), idxTensor}));
        }
    }
}

void AlphaCROWNAnalysis::restoreBestAlphas()
{
    log("restoreBestAlphas() - Restoring all alpha parameters to best found values");

    auto& currentStorage = _alphaByNodeStart;
    auto& bestStorage = _bestAlphaByNodeStart;

    for (auto& [nodeIdx, perStart] : currentStorage) {
        for (auto& [startKey, ap] : perStart) {
            auto& bestAp = bestStorage[nodeIdx][startKey];
            if (ap.alpha.defined() && bestAp.alpha.defined()
                && ap.alpha.sizes() == bestAp.alpha.sizes()) {
                ap.alpha.data().copy_(bestAp.alpha.data());
            } else {
                if (ap.alpha.defined()) {
                    bestAp.alpha = ap.alpha.detach().clone();
                }
            }
        }
    }
}

void AlphaCROWNAnalysis::snapshotBestIntermediateBounds()
{
    unsigned numNodesUpdated = 0;
    unsigned numNodesNew = 0;

    unsigned numNodes = _crownAnalysis->getNumNodes();
    for (unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
        auto node = _crownAnalysis->getNode(nodeIdx);
        if (!node) continue;

        bool isPreActivation = false;

        for (const auto& [actIdx, actNode] : _optimizableNodes) {
            (void)actNode;
            auto deps = _torchModel->getDependencies(actIdx);
            for (unsigned depIdx : deps) {
                if (depIdx == nodeIdx) {
                    isPreActivation = true;
                    break;
                }
            }
            if (isPreActivation) break;
        }

        if (!isPreActivation) continue;

        if (_crownAnalysis->hasConcreteBounds(nodeIdx)) {
            torch::Tensor currentLower = _crownAnalysis->getConcreteLowerBound(nodeIdx).detach();
            torch::Tensor currentUpper = _crownAnalysis->getConcreteUpperBound(nodeIdx).detach();

            auto it = _bestIntermediateBounds.find(nodeIdx);
            if (it != _bestIntermediateBounds.end()) {
                // element-wise merge: tighter lower = max, tighter upper = min
                torch::Tensor& storedLower = it->second.first;
                torch::Tensor& storedUpper = it->second.second;

                storedLower = torch::max(storedLower, currentLower).detach().clone();
                storedUpper = torch::min(storedUpper, currentUpper).detach().clone();
                numNodesUpdated++;
            } else {
                _bestIntermediateBounds[nodeIdx] = std::make_pair(
                    currentLower.detach().clone(),
                    currentUpper.detach().clone());
                numNodesNew++;
            }
        }
    }

    if (numNodesNew > 0 || numNodesUpdated > 0) {
        log(Stringf("snapshotBestIntermediateBounds() - Element-wise best: %u nodes updated, %u nodes new",
                    numNodesUpdated, numNodesNew));
    }
}

void AlphaCROWNAnalysis::restoreBestIntermediateBounds()
{
    if (_bestIntermediateBounds.empty()) {
        return;
    }

    log("restoreBestIntermediateBounds() - Seeding best bounds into nodes");

    unsigned numRestored = 0;
    for (const auto& [nodeIdx, bounds] : _bestIntermediateBounds) {
        auto node = _crownAnalysis->getNode(nodeIdx);
        if (node) {
            auto lower = bounds.first.detach();
            auto upper = bounds.second.detach();
            lower.requires_grad_(false);
            upper.requires_grad_(false);
            node->setBounds(lower, upper);
            numRestored++;
        }
    }

    log(Stringf("restoreBestIntermediateBounds() - Restored bounds for %u nodes", numRestored));
}

torch::Tensor AlphaCROWNAnalysis::computeLoss(const torch::Tensor& lowerBounds,
                                              const torch::Tensor& upperBounds,
                                              bool optimizeLower,
                                              c10::optional<torch::Tensor> stop_mask_opt /* (batch,1) bool */) {

    (void)stop_mask_opt;
    auto options = lowerBounds.defined() ? lowerBounds.options()
                                         : (upperBounds.defined() ? upperBounds.options()
                                                                  : torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor loss_per_elem;

    if (optimizeLower) {
        if (!lowerBounds.defined() || lowerBounds.numel() == 0) {
            return torch::zeros({1}, options);
        }
        if (lowerBounds.dim() == 1) {
            loss_per_elem = -lowerBounds.sum().unsqueeze(0);
        } else {
            auto l = lowerBounds.sum(/*dim=*/1, /*keepdim=*/true);
            loss_per_elem = -l;
        }
    } else {
        if (!upperBounds.defined() || upperBounds.numel() == 0) {
            return torch::zeros({1}, options);
        }
        if (upperBounds.dim() == 1) {
            loss_per_elem = upperBounds.sum().unsqueeze(0);
        } else {
            auto u = upperBounds.sum(/*dim=*/1, /*keepdim=*/true);
            loss_per_elem = u;
        }
    }

    auto loss = loss_per_elem.sum();

    return loss;
}

std::vector<int> AlphaCROWNAnalysis::findImprovedIndices(const torch::Tensor& currentBounds, const torch::Tensor& bestBounds, bool isLowerBound){

    std::vector<int> improvedIndices;

    if ( !currentBounds.defined() || !bestBounds.defined() || currentBounds.sizes() != bestBounds.sizes() )
    {
        return improvedIndices;
    }

    torch::Tensor improved = isLowerBound ? (currentBounds > bestBounds)
                                          : (currentBounds < bestBounds);

    if (!improved.defined() || improved.numel() == 0) {
        return improvedIndices;
    }

    torch::Tensor perSpec;
    if (improved.dim() == 0) {
        if (improved.item<bool>()) {
            improvedIndices.push_back(0);
        }
        return improvedIndices;
    } else if (improved.dim() == 1) {
        perSpec = improved;
    } else {
        perSpec = improved;
        for (int d = 0; d < improved.dim() - 1; ++d) {
            perSpec = perSpec.any(0);
        }
    }

    auto improvedMask = perSpec.nonzero().flatten().to(torch::kInt32).cpu().contiguous();
    if (improvedMask.numel() > 0) {
        const int* data = improvedMask.data_ptr<int>();
        int64_t n = improvedMask.size(0);
        improvedIndices.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            improvedIndices.push_back(data[i]);
        }
    }

    return improvedIndices;
}

bool AlphaCROWNAnalysis::shouldPerformOptimization() const
{
    if ( !_alphaEnabled )
    {
        return false;
    }

    if ( !_initialized && _optimizableNodes.empty() )
    {
        unsigned numNodes = _crownAnalysis->getNumNodes();
        for ( unsigned i = 0; i < numNodes; ++i )
        {
            auto node = _crownAnalysis->getNode(i);
            if ( node && std::dynamic_pointer_cast<BoundedAlphaOptimizeNode>(node) )
            {
                return true;
            }
        }
        return false;
    }

    return !_optimizableNodes.empty();
}

AlphaCROWNAnalysis::AlphaResult AlphaCROWNAnalysis::getAlphaForNodeAllSpecs(
    unsigned nodeIndex,
    const std::string& startKey,
    int specDim,
    int outDim,
    const torch::Tensor& input_lb,
    const torch::Tensor& input_ub)
{

    auto& ap = ensureAlphaFor(nodeIndex, startKey, specDim, outDim, input_lb, input_ub);

    AlphaResult result;
    result.numUnstable = ap.numUnstable;
    result.outDim = ap.outDim;
    result.unstableMask = ap.unstableMask;
    result.unstableIndices = ap.unstableIndices;
    result.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;

    // squeeze(2) drops batch dim, preserving gradient connection
    result.alpha = ap.alpha.squeeze(2);

    return result;
}

void AlphaCROWNAnalysis::setOptimizationStage(const std::string& stage)
{
    _optimizationStage = stage;

    for (auto& nodePair : _optimizableNodes) {
        auto node = nodePair.second;
        node->setOptimizationStage(stage);
    }

    log(Stringf("setOptimizationStage() - Set optimization stage to '%s'", stage.c_str()));
}

unsigned AlphaCROWNAnalysis::getNumOptimizableNodes() const
{
    return static_cast<unsigned>(_optimizableNodes.size());
}

std::vector<unsigned> AlphaCROWNAnalysis::getOptimizableNodeIndices() const
{
    std::vector<unsigned> indices;
    indices.reserve(_optimizableNodes.size());

    for (const auto& nodePair : _optimizableNodes) {
        indices.push_back(nodePair.first);
    }

    return indices;
}

void AlphaCROWNAnalysis::clipAlphaParameters()
{
    log("clipAlphaParameters() - Clipping alpha parameters to valid ranges [0, 1]");

    // .data().clamp_() modifies in-place without breaking autograd;
    // set_data() or new tensors cause version mismatch with squeeze() views
    torch::NoGradGuard no_grad;

    auto shouldClip01 = [&](unsigned nodeIdx) -> bool {
        auto node = _crownAnalysis->getNode(nodeIdx);
        // sigmoid uses tangent-point alpha, not [0,1] slopes
        return !(node && node->getNodeType() == NodeType::SIGMOID);
    };

    for (auto& [nodeIdx, perStart] : _alphaByNodeStart) {
        for (auto& [startKey, ap] : perStart) {
            (void)startKey;
            if (!shouldClip01(nodeIdx)) continue;
            ap.alpha.data().clamp_(0.0f, 1.0f);
        }
    }

    log("clipAlphaParameters() - Alpha parameter clipping completed");
}

void AlphaCROWNAnalysis::resetForOptimization()
{
    log("resetForOptimization() - Resetting analysis state for optimization");

    setOptimizationStage("opt");

    for (auto& [nodeIdx, perStart] : _alphaByNodeStart) {
        for (auto& [startKey, ap] : perStart) {
            ap.alpha.requires_grad_(true);
        }
    }

    log("resetForOptimization() - Analysis state reset for optimization");
}

void AlphaCROWNAnalysis::resetAlphasFromCROWNSlopes(bool isLower)
{
    (void)isLower;

    log("resetAlphasFromCROWNSlopes() - Re-initializing alpha parameters from CROWN slopes");

    _alphaByNodeStart.clear();
    _bestAlphaByNodeStart.clear();
    _crownAnalysis->clearAlphaStartCache();

    log("resetAlphasFromCROWNSlopes() - Cleared alpha storage, alphas will be re-created lazily");
}

CROWNAnalysis* AlphaCROWNAnalysis::getCROWNAnalysis() const
{
    return _crownAnalysis.get();
}

TorchModel* AlphaCROWNAnalysis::getTorchModel() const
{
    return _torchModel;
}

void AlphaCROWNAnalysis::updateFromConfig()
{
    _alphaEnabled = (LunaConfiguration::ANALYSIS_METHOD == LunaConfiguration::AnalysisMethod::AlphaCROWN);
    _iteration = LunaConfiguration::ALPHA_ITERATIONS;
    _learningRate = LunaConfiguration::ALPHA_LR;

    log(Stringf("updateFromConfig() - Updated from LunaConfiguration: enable=%s, iterations=%u, lr=%.3f",
               _alphaEnabled ? "true" : "false", _iteration, _learningRate));
}

void AlphaCROWNAnalysis::log(const String& message)
{
    if (LunaConfiguration::NETWORK_LEVEL_REASONER_LOGGING || LunaConfiguration::VERBOSE) {
        printf("AlphaCROWNAnalysis: %s\n", message.ascii());
    }
}

void AlphaCROWNAnalysis::preserveIntermediateBoundsForNextOptimization()
{
    _reuseIntermediateBounds = !_bestIntermediateBounds.empty();
    if (_reuseIntermediateBounds) {
        log("preserveIntermediateBoundsForNextOptimization() - Intermediate bounds preserved for reuse");
    }
}

void AlphaCROWNAnalysis::seedAlphas(
    const std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>>& alphas)
{
    for (auto& [nodeIdx, perStart] : alphas) {
        for (auto& [startKey, ap] : perStart) {
            AlphaParameters copy;
            copy.alpha = ap.alpha.defined() ? ap.alpha.detach().clone() : torch::Tensor();
            if (copy.alpha.defined())
                copy.alpha.set_requires_grad(true);
            copy.unstableMask = ap.unstableMask.defined() ? ap.unstableMask.detach().clone() : torch::Tensor();
            copy.unstableIndices = ap.unstableIndices.defined() ? ap.unstableIndices.detach().clone() : torch::Tensor();
            copy.specDim = ap.specDim;
            copy.batchDim = ap.batchDim;
            copy.outDim = ap.outDim;
            copy.numUnstable = ap.numUnstable;
            copy.requiresGrad = true;
            copy.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;
            _alphaByNodeStart[nodeIdx][startKey] = std::move(copy);
        }
    }
    _bestAlphaByNodeStart = _alphaByNodeStart;
    if (_optimizableNodes.empty())
        prepareOptimizableActivations();
    _initialized = true;
    log("seedAlphas() - Seeded alpha parameters from BaB state");
}

void AlphaCROWNAnalysis::seedIntermediateBounds(
    const std::unordered_map<unsigned, std::pair<torch::Tensor, torch::Tensor>>& bounds)
{
    for (auto& [nodeIdx, pair] : bounds) {
        _bestIntermediateBounds[nodeIdx] = {
            pair.first.detach().clone(),
            pair.second.detach().clone()
        };
    }
    if (!LunaConfiguration::RECOMPUTE_INTERMEDIATE_BOUNDS) {
        restoreBestIntermediateBounds();
    } else if (LunaConfiguration::STABILIZE_INTERMEDIATE_BOUNDS) {
        for (auto& [nodeIdx, bp] : _bestIntermediateBounds) {
            _referenceBounds[nodeIdx] = {bp.first.detach().clone(), bp.second.detach().clone()};
        }
        _crownAnalysis->setReferenceBounds(_referenceBounds);
    }
    log("seedIntermediateBounds() - Seeded intermediate bounds from BaB state");
}

std::unordered_map<unsigned,
    std::unordered_map<std::string, AlphaParameters>>
AlphaCROWNAnalysis::extractAlphas() const
{
    const auto& source = _bestAlphaByNodeStart.empty()
        ? _alphaByNodeStart : _bestAlphaByNodeStart;
    std::unordered_map<unsigned,
        std::unordered_map<std::string, AlphaParameters>> result;
    for (auto& [nodeIdx, perStart] : source) {
        for (auto& [startKey, ap] : perStart) {
            AlphaParameters copy;
            copy.alpha = ap.alpha.defined() ? ap.alpha.detach().clone() : torch::Tensor();
            copy.unstableMask = ap.unstableMask.defined() ? ap.unstableMask.detach().clone() : torch::Tensor();
            copy.unstableIndices = ap.unstableIndices.defined() ? ap.unstableIndices.detach().clone() : torch::Tensor();
            copy.specDim = ap.specDim;
            copy.batchDim = ap.batchDim;
            copy.outDim = ap.outDim;
            copy.numUnstable = ap.numUnstable;
            copy.requiresGrad = ap.requiresGrad;
            copy.hasSpecDefaultSlot = ap.hasSpecDefaultSlot;
            result[nodeIdx][startKey] = std::move(copy);
        }
    }
    return result;
}

} // namespace NLR
