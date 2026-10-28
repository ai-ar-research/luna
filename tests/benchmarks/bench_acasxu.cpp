#include <benchmark/benchmark.h>
#include "src/engine/TorchModel.h"
#include "fixtures/model_builders.h"
#include <torch/torch.h>

using namespace NLR;
using namespace test;

static void BM_ACASXu_CROWN( benchmark::State &state )
{
    std::string onnxPath = RESOURCES_DIR "/onnx/acasxu/ACASXU_experimental_v2a_1_1.onnx";
    std::string vnnlibPath = RESOURCES_DIR "/onnx/vnnlib/prop_1.vnnlib";

    std::shared_ptr<TorchModel> model;
    try
    {
        model = ModelBuilder::loadFromONNXWithVNNLib( onnxPath, vnnlibPath );
    }
    catch ( ... )
    {
        state.SkipWithError( "ACAS Xu model not available" );
        return;
    }

    for ( auto _ : state )
    {
        model->clearConcreteBounds();
        auto bounds = model->runCROWN();
        benchmark::DoNotOptimize( bounds );
    }
}

BENCHMARK( BM_ACASXu_CROWN )->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
