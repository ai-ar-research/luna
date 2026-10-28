#include <benchmark/benchmark.h>
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

static void BM_AlphaCROWN_Iterations( benchmark::State &state )
{
    unsigned iterations = state.range( 0 );
    LunaConfiguration::ALPHA_ITERATIONS = iterations;

    auto model = ModelBuilder::createMLP( 5, { 10, 10 }, 3, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 5 }, 0.0, 1.0, 0.1 );
    model->setInputBounds( inputBounds );

    for ( auto _ : state )
    {
        model->clearConcreteBounds();
        auto bounds = model->runAlphaCROWN( true, false );
        benchmark::DoNotOptimize( bounds );
    }
}

BENCHMARK( BM_AlphaCROWN_Iterations )
    ->Arg( 5 )
    ->Arg( 10 )
    ->Arg( 20 )
    ->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
