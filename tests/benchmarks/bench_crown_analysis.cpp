#include <benchmark/benchmark.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include "fixtures/test_utils.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

static void BM_CROWN_MLP_Depth( benchmark::State &state )
{
    unsigned depth = state.range( 0 );
    unsigned width = state.range( 1 );

    std::vector<unsigned> hiddenSizes( depth, width );
    auto model = ModelBuilder::createMLP( 10, hiddenSizes, 5, true, false );
    auto inputBounds = BoundGenerator::randomBounds( { 1, 10 }, 0.0, 1.0, 0.1 );
    model->setInputBounds( inputBounds );

    for ( auto _ : state )
    {
        model->clearConcreteBounds();
        auto bounds = model->runCROWN();
        benchmark::DoNotOptimize( bounds );
    }

    state.counters["neurons"] = depth * width;
}

BENCHMARK( BM_CROWN_MLP_Depth )
    ->Args( { 2, 16 } )
    ->Args( { 4, 16 } )
    ->Args( { 8, 16 } )
    ->Args( { 2, 64 } )
    ->Args( { 4, 64 } )
    ->Args( { 8, 64 } )
    ->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
