#include <benchmark/benchmark.h>
#include "src/engine/TorchModel.h"
#include "src/engine/CROWNAnalysis.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

static void BM_IBP_Analysis( benchmark::State &state )
{
    auto model = ModelBuilder::createMLP( 10, { 20, 20 }, 5, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 10 }, 0.0, 1.0, 0.1 );
    model->setInputBounds( inputBounds );

    NLR::CROWNAnalysis analysis( model.get() );
    analysis.setInputBounds( inputBounds );

    for ( auto _ : state )
    {
        analysis.computeIBPBounds();
        auto bounds = analysis.getOutputIBPBounds();
        benchmark::DoNotOptimize( bounds );
    }
}

static void BM_CROWN_Analysis( benchmark::State &state )
{
    auto model = ModelBuilder::createMLP( 10, { 20, 20 }, 5, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 10 }, 0.0, 1.0, 0.1 );
    model->setInputBounds( inputBounds );

    for ( auto _ : state )
    {
        model->clearConcreteBounds();
        auto bounds = model->runCROWN();
        benchmark::DoNotOptimize( bounds );
    }
}

BENCHMARK( BM_IBP_Analysis )->Unit( benchmark::kMillisecond );
BENCHMARK( BM_CROWN_Analysis )->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
