#include "src/engine/AlphaCROWNAnalysis.h"
#include "src/engine/CROWNAnalysis.h"
#include "src/engine/TorchModel.h"
#include "src/configuration/LunaConfiguration.h"
#include "src/engine/nodes/BoundedInputNode.h"
#include "src/engine/nodes/BoundedLinearNode.h"
#include "src/engine/nodes/BoundedReLUNode.h"
#include "src/common/Vector.h"
#include "src/common/Map.h"

#include <torch/torch.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <utility>

// These runners historically used Marabou's `Variable` type. In this standalone LIRPA build,
// we only need an integer identifier for bookkeeping.
using Variable = unsigned;

NLR::TorchModel *createSinglePathwayNetwork()
{
    using namespace NLR;

    std::vector<std::shared_ptr<BoundedTorchNode>> nodes;
    std::vector<std::vector<Variable>> marabouVars;
    std::vector<unsigned> inputIndices;
    std::map<unsigned, std::vector<Variable>> neuronToMarabouMap;
    std::map<unsigned, std::vector<unsigned>> dependencies;

    // Node indices:
    // 0: input(2)
    // 1: linear1(2->1, no bias, W = [1, 1])
    // 2: relu1(1)
    // 3: linear2(1->1, w=0.5, b=-0.2)
    // 4: relu2(1)
    // 5: linear3(1->1, w=1.0, b=0.5)
    // Output index = 5

    // Input
    auto inputNode = std::make_shared<BoundedInputNode>( 0, 2, "input" );
    inputNode->setNodeIndex( 0 );
    nodes.push_back( inputNode );

    // Linear 1: (2 -> 1) with NO BIAS to match Python
    // If your BoundedLinearNode requires a bias tensor, you can instead set bias(true) and assign
    // zeros.
    torch::nn::Linear linear1( torch::nn::LinearOptions( 2, 1 ).bias( false ) );
    linear1->weight = torch::tensor( { { 1.0, 1.0 } }, torch::kFloat64 );
    auto linearNode1 = std::make_shared<BoundedLinearNode>( linear1, 1.0f, "linear1" );
    linearNode1->setNodeIndex( 1 );
    nodes.push_back( linearNode1 );

    // ReLU 1
    torch::nn::ReLU relu1( torch::nn::ReLUOptions{} );
    auto reluNode1 = std::make_shared<BoundedReLUNode>( relu1, "relu1" );
    reluNode1->setNodeIndex( 2 );
    nodes.push_back( reluNode1 );

    // Linear 2: (1 -> 1), w=0.5, b=-0.2
    torch::nn::Linear linear2( torch::nn::LinearOptions( 1, 1 ) );
    linear2->weight = torch::tensor( { { 0.5 } }, torch::kFloat64 );
    linear2->bias = torch::tensor( { -0.2 }, torch::kFloat64 );
    auto linearNode2 = std::make_shared<BoundedLinearNode>( linear2, 1.0f, "linear2" );
    linearNode2->setNodeIndex( 3 );
    nodes.push_back( linearNode2 );

    // ReLU 2
    torch::nn::ReLU relu2( torch::nn::ReLUOptions{} );
    auto reluNode2 = std::make_shared<BoundedReLUNode>( relu2, "relu2" );
    reluNode2->setNodeIndex( 4 );
    nodes.push_back( reluNode2 );

    // Linear 3: (1 -> 1), w=1.0, b=0.5
    torch::nn::Linear linear3( torch::nn::LinearOptions( 1, 1 ) );
    linear3->weight = torch::tensor( { { 1.0 } }, torch::kFloat64 );
    linear3->bias = torch::tensor( { 0.5 }, torch::kFloat64 );
    auto linearNode3 = std::make_shared<BoundedLinearNode>( linear3, 1.0f, "linear3" );
    linearNode3->setNodeIndex( 5 );
    nodes.push_back( linearNode3 );

    // Dependencies
    dependencies[0] = {};    // input has no deps
    dependencies[1] = { 0 }; // linear1 <- input
    dependencies[2] = { 1 }; // relu1   <- linear1
    dependencies[3] = { 2 }; // linear2 <- relu1
    dependencies[4] = { 3 }; // relu2   <- linear2
    dependencies[5] = { 4 }; // linear3 <- relu2

    inputIndices.push_back( 0 );
    unsigned outputIndex = 5;

    // Variable mapping
    unsigned varIndex = 0;

    // Node 0 (input): 2 vars
    std::vector<Variable> inputVars;
    for ( int i = 0; i < 2; ++i )
        inputVars.push_back( Variable( varIndex++ ) );
    marabouVars.push_back( inputVars );
    neuronToMarabouMap[0] = inputVars;

    // Node 1 (linear1): 1 var
    std::vector<Variable> l1Vars{ Variable( varIndex++ ) };
    marabouVars.push_back( l1Vars );
    neuronToMarabouMap[1] = l1Vars;

    // Node 2 (relu1): 1 var
    std::vector<Variable> r1Vars{ Variable( varIndex++ ) };
    marabouVars.push_back( r1Vars );
    neuronToMarabouMap[2] = r1Vars;

    // Node 3 (linear2): 1 var
    std::vector<Variable> l2Vars{ Variable( varIndex++ ) };
    marabouVars.push_back( l2Vars );
    neuronToMarabouMap[3] = l2Vars;

    // Node 4 (relu2): 1 var
    std::vector<Variable> r2Vars{ Variable( varIndex++ ) };
    marabouVars.push_back( r2Vars );
    neuronToMarabouMap[4] = r2Vars;

    // Node 5 (linear3): 1 var
    std::vector<Variable> l3Vars{ Variable( varIndex++ ) };
    marabouVars.push_back( l3Vars );
    neuronToMarabouMap[5] = l3Vars;

    // Sizes
    inputNode->setInputSize( 2 );
    inputNode->setOutputSize( 2 );

    linearNode1->setInputSize( 2 );
    linearNode1->setOutputSize( 1 );

    reluNode1->setInputSize( 1 );
    reluNode1->setOutputSize( 1 );

    linearNode2->setInputSize( 1 );
    linearNode2->setOutputSize( 1 );

    reluNode2->setInputSize( 1 );
    reluNode2->setOutputSize( 1 );

    linearNode3->setInputSize( 1 );
    linearNode3->setOutputSize( 1 );

    // Convert to Marabou containers
    Vector<std::shared_ptr<BoundedTorchNode>> marabouNodes;
    for ( const auto &n : nodes )
        marabouNodes.append( n );

    Vector<Vector<Variable>> marabouVarsVec;
    for ( const auto &vs : marabouVars )
    {
        Vector<Variable> inner;
        for ( const auto &v : vs )
            inner.append( v );
        marabouVarsVec.append( inner );
    }

    Vector<unsigned> marabouInputIndices;
    for ( auto idx : inputIndices )
        marabouInputIndices.append( idx );

    Map<unsigned, Vector<Variable>> marabouNeuronMap;
    for ( const auto &kv : neuronToMarabouMap )
    {
        Vector<Variable> inner;
        for ( const auto &v : kv.second )
            inner.append( v );
        marabouNeuronMap[kv.first] = inner;
    }

    Map<unsigned, Vector<unsigned>> marabouDependencies;
    for ( const auto &kv : dependencies )
    {
        Vector<unsigned> inner;
        for ( auto d : kv.second )
            inner.append( d );
        marabouDependencies[kv.first] = inner;
    }

    // Build model
    auto *model =
        new TorchModel( marabouNodes, marabouInputIndices, outputIndex, marabouDependencies );

    return model;
}

NLR::TorchModel *createDeepPolyNetwork()
{

    std::vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    std::vector<std::vector<Variable>> marabouVars;
    std::vector<unsigned> inputIndices;
    std::map<unsigned, std::vector<Variable>> neuronToMarabouMap;
    std::map<unsigned, std::vector<unsigned>> dependencies;

    // Model architecture from alpha.py DeepPolyNetwork:
    // Input(2) -> Linear(2,2) -> ReLU -> Linear(2,2) -> ReLU -> Linear(2,2) -> Output

    // Create input node (2 input neurons)
    auto inputNode = std::make_shared<NLR::BoundedInputNode>( 0, 2, "input" );
    inputNode->setNodeIndex( 0 );
    nodes.push_back( inputNode );

    // Create first linear layer (2 -> 2 neurons)
    torch::nn::Linear linear1( torch::nn::LinearOptions( 2, 2 ) );
    // Set weights according to alpha.py: [[1,1],[1,-1]]
    linear1->weight = torch::tensor( { { 1.0, 1.0 }, { 1.0, -1.0 } }, torch::kFloat64 );
    linear1->bias = torch::zeros( { 2 }, torch::kFloat64 );
    auto linearNode1 = std::make_shared<NLR::BoundedLinearNode>( linear1, 1.0f, "linear1" );
    linearNode1->setNodeIndex( 1 );
    nodes.push_back( linearNode1 );

    // Create first ReLU activation
    torch::nn::ReLU relu1( torch::nn::ReLUOptions{} );
    auto reluNode1 = std::make_shared<NLR::BoundedReLUNode>( relu1, "relu1" );
    reluNode1->setNodeIndex( 2 );
    nodes.push_back( reluNode1 );

    // Create second linear layer (2 -> 2 neurons)
    torch::nn::Linear linear2( torch::nn::LinearOptions( 2, 2 ) );
    // Set weights according to alpha.py: [[1,1],[1,-1]]
    linear2->weight = torch::tensor( { { 1.0, 1.0 }, { 1.0, -1.0 } }, torch::kFloat64 );
    linear2->bias = torch::zeros( { 2 }, torch::kFloat64 );
    auto linearNode2 = std::make_shared<NLR::BoundedLinearNode>( linear2, 1.0f, "linear2" );
    linearNode2->setNodeIndex( 3 );
    nodes.push_back( linearNode2 );

    // Create second ReLU activation
    torch::nn::ReLU relu2( torch::nn::ReLUOptions{} );
    auto reluNode2 = std::make_shared<NLR::BoundedReLUNode>( relu2, "relu2" );
    reluNode2->setNodeIndex( 4 );
    nodes.push_back( reluNode2 );

    // Create third linear layer (2 -> 2 neurons)
    torch::nn::Linear linear3( torch::nn::LinearOptions( 2, 2 ) );
    // Set weights according to alpha.py: [[1,1],[0,1]] with bias [1,0]
    linear3->weight = torch::tensor( { { 1.0, 1.0 }, { 0.0, 1.0 } }, torch::kFloat64 );
    linear3->bias = torch::tensor( { 1.0, 0.0 }, torch::kFloat64 );
    auto linearNode3 = std::make_shared<NLR::BoundedLinearNode>( linear3, 1.0f, "linear3" );
    linearNode3->setNodeIndex( 5 );
    nodes.push_back( linearNode3 );

    // Set up dependencies: input -> linear1 -> relu1 -> linear2 -> relu2 -> linear3
    std::vector<unsigned> inputDeps;
    dependencies[0] = inputDeps;

    std::vector<unsigned> linear1Deps;
    linear1Deps.push_back( 0 );
    dependencies[1] = linear1Deps;

    std::vector<unsigned> relu1Deps;
    relu1Deps.push_back( 1 );
    dependencies[2] = relu1Deps;

    std::vector<unsigned> linear2Deps;
    linear2Deps.push_back( 2 );
    dependencies[3] = linear2Deps;

    std::vector<unsigned> relu2Deps;
    relu2Deps.push_back( 3 );
    dependencies[4] = relu2Deps;

    std::vector<unsigned> linear3Deps;
    linear3Deps.push_back( 4 );
    dependencies[5] = linear3Deps;

    // Set input indices
    inputIndices.push_back( 0 );

    // Set output index (last node)
    unsigned outputIndex = 5;

    // Set up variable mapping for Marabou
    // Node 0 (input): variables 0, 1
    std::vector<Variable> inputVars;
    inputVars.push_back( Variable( 0 ) );
    inputVars.push_back( Variable( 1 ) );
    marabouVars.push_back( inputVars );
    neuronToMarabouMap[0] = inputVars;

    // Node 1 (linear1): variables 2, 3
    std::vector<Variable> linear1Vars;
    linear1Vars.push_back( Variable( 2 ) );
    linear1Vars.push_back( Variable( 3 ) );
    marabouVars.push_back( linear1Vars );
    neuronToMarabouMap[1] = linear1Vars;

    // Node 2 (relu1): variables 4, 5
    std::vector<Variable> relu1Vars;
    relu1Vars.push_back( Variable( 4 ) );
    relu1Vars.push_back( Variable( 5 ) );
    marabouVars.push_back( relu1Vars );
    neuronToMarabouMap[2] = relu1Vars;

    // Node 3 (linear2): variables 6, 7
    std::vector<Variable> linear2Vars;
    linear2Vars.push_back( Variable( 6 ) );
    linear2Vars.push_back( Variable( 7 ) );
    marabouVars.push_back( linear2Vars );
    neuronToMarabouMap[3] = linear2Vars;

    // Node 4 (relu2): variables 8, 9
    std::vector<Variable> relu2Vars;
    relu2Vars.push_back( Variable( 8 ) );
    relu2Vars.push_back( Variable( 9 ) );
    marabouVars.push_back( relu2Vars );
    neuronToMarabouMap[4] = relu2Vars;

    // Node 5 (linear3): variables 10, 11
    std::vector<Variable> linear3Vars;
    linear3Vars.push_back( Variable( 10 ) );
    linear3Vars.push_back( Variable( 11 ) );
    marabouVars.push_back( linear3Vars );
    neuronToMarabouMap[5] = linear3Vars;

    // Set sizes for nodes
    inputNode->setInputSize( 2 );
    inputNode->setOutputSize( 2 );

    linearNode1->setInputSize( 2 );
    linearNode1->setOutputSize( 2 );

    reluNode1->setInputSize( 2 );
    reluNode1->setOutputSize( 2 );

    linearNode2->setInputSize( 2 );
    linearNode2->setOutputSize( 2 );

    reluNode2->setInputSize( 2 );
    reluNode2->setOutputSize( 2 );

    linearNode3->setInputSize( 2 );
    linearNode3->setOutputSize( 2 );

    // Convert std containers to Marabou containers
    Vector<std::shared_ptr<NLR::BoundedTorchNode>> marabouNodes;
    for ( const auto &node : nodes )
    {
        marabouNodes.append( node );
    }

    Vector<Vector<Variable>> marabouVarsVec;
    for ( const auto &vars : marabouVars )
    {
        Vector<Variable> marabouVarsInner;
        for ( const auto &var : vars )
        {
            marabouVarsInner.append( var );
        }
        marabouVarsVec.append( marabouVarsInner );
    }

    Vector<unsigned> marabouInputIndices;
    for ( const auto &idx : inputIndices )
    {
        marabouInputIndices.append( idx );
    }

    Map<unsigned, Vector<Variable>> marabouNeuronMap;
    for ( const auto &pair : neuronToMarabouMap )
    {
        Vector<Variable> marabouVarsInner;
        for ( const auto &var : pair.second )
        {
            marabouVarsInner.append( var );
        }
        marabouNeuronMap[pair.first] = marabouVarsInner;
    }

    Map<unsigned, Vector<unsigned>> marabouDependencies;
    for ( const auto &pair : dependencies )
    {
        Vector<unsigned> marabouDeps;
        for ( const auto &dep : pair.second )
        {
            marabouDeps.append( dep );
        }
        marabouDependencies[pair.first] = marabouDeps;
    }

    // Create and return the TorchModel
    NLR::TorchModel *model =
        new NLR::TorchModel( marabouNodes, marabouInputIndices, outputIndex, marabouDependencies );

    return model;
}

NLR::TorchModel *createDeeperNetwork()
{

    std::vector<std::shared_ptr<NLR::BoundedTorchNode>> nodes;
    std::vector<std::vector<Variable>> marabouVars;
    std::vector<unsigned> inputIndices;
    std::map<unsigned, std::vector<Variable>> neuronToMarabouMap;
    std::map<unsigned, std::vector<unsigned>> dependencies;

    // Model architecture from alpha.py DeeperNetwork:
    // Input(2) -> Linear(2,4) -> ReLU -> Linear(4,4) -> ReLU -> Linear(4,3) -> ReLU -> Linear(3,1)

    // Create input node (2 input neurons)
    auto inputNode = std::make_shared<NLR::BoundedInputNode>( 0, 2, "input" );
    inputNode->setNodeIndex( 0 );
    nodes.push_back( inputNode );

    // Create first linear layer (2 -> 4 neurons)
    torch::nn::Linear fc1( torch::nn::LinearOptions( 2, 4 ) );
    fc1->weight = torch::tensor( { { 0.5, -0.5 }, { 0.8, 0.2 }, { -0.3, 0.7 }, { 0.6, -0.4 } },
                                 torch::kFloat64 );
    fc1->bias = torch::tensor( { 0.1, -0.2, 0.3, -0.1 }, torch::kFloat64 );
    auto linearNode1 = std::make_shared<NLR::BoundedLinearNode>( fc1, 1.0f, "fc1" );
    linearNode1->setNodeIndex( 1 );
    nodes.push_back( linearNode1 );

    // Create first ReLU activation
    torch::nn::ReLU relu1( torch::nn::ReLUOptions{} );
    auto reluNode1 = std::make_shared<NLR::BoundedReLUNode>( relu1, "relu1" );
    reluNode1->setNodeIndex( 2 );
    nodes.push_back( reluNode1 );

    // Create second linear layer (4 -> 4 neurons)
    torch::nn::Linear fc2( torch::nn::LinearOptions( 4, 4 ) );
    fc2->weight = torch::tensor( { { 0.4, -0.2, 0.3, 0.1 },
                                   { -0.1, 0.5, -0.3, 0.2 },
                                   { 0.2, 0.3, -0.4, 0.5 },
                                   { -0.3, 0.1, 0.2, -0.6 } },
                                 torch::kFloat64 );
    fc2->bias = torch::tensor( { 0.05, -0.1, 0.15, -0.05 }, torch::kFloat64 );
    auto linearNode2 = std::make_shared<NLR::BoundedLinearNode>( fc2, 1.0f, "fc2" );
    linearNode2->setNodeIndex( 3 );
    nodes.push_back( linearNode2 );

    // Create second ReLU activation
    torch::nn::ReLU relu2( torch::nn::ReLUOptions{} );
    auto reluNode2 = std::make_shared<NLR::BoundedReLUNode>( relu2, "relu2" );
    reluNode2->setNodeIndex( 4 );
    nodes.push_back( reluNode2 );

    // Create third linear layer (4 -> 3 neurons)
    torch::nn::Linear fc3( torch::nn::LinearOptions( 4, 3 ) );
    fc3->weight = torch::tensor(
        { { 0.3, -0.4, 0.2, 0.5 }, { -0.2, 0.3, -0.1, 0.4 }, { 0.4, 0.1, -0.5, -0.2 } },
        torch::kFloat64 );
    fc3->bias = torch::tensor( { 0.2, -0.1, 0.1 }, torch::kFloat64 );
    auto linearNode3 = std::make_shared<NLR::BoundedLinearNode>( fc3, 1.0f, "fc3" );
    linearNode3->setNodeIndex( 5 );
    nodes.push_back( linearNode3 );

    // Create third ReLU activation
    torch::nn::ReLU relu3( torch::nn::ReLUOptions{} );
    auto reluNode3 = std::make_shared<NLR::BoundedReLUNode>( relu3, "relu3" );
    reluNode3->setNodeIndex( 6 );
    nodes.push_back( reluNode3 );

    // Create fourth linear layer (3 -> 1 neuron)
    torch::nn::Linear fc4( torch::nn::LinearOptions( 3, 1 ) );
    fc4->weight = torch::tensor( { { 0.7, -0.3, 0.4 } }, torch::kFloat64 );
    fc4->bias = torch::tensor( { 0.0 }, torch::kFloat64 );
    auto linearNode4 = std::make_shared<NLR::BoundedLinearNode>( fc4, 1.0f, "fc4" );
    linearNode4->setNodeIndex( 7 );
    nodes.push_back( linearNode4 );

    // Set up dependencies
    dependencies[0] = std::vector<unsigned>{};
    dependencies[1] = std::vector<unsigned>{ 0 };
    dependencies[2] = std::vector<unsigned>{ 1 };
    dependencies[3] = std::vector<unsigned>{ 2 };
    dependencies[4] = std::vector<unsigned>{ 3 };
    dependencies[5] = std::vector<unsigned>{ 4 };
    dependencies[6] = std::vector<unsigned>{ 5 };
    dependencies[7] = std::vector<unsigned>{ 6 };

    inputIndices.push_back( 0 );
    unsigned outputIndex = 7;

    // Set up variable mapping
    unsigned varIndex = 0;

    // Node 0 (input): 2 variables
    std::vector<Variable> inputVars;
    for ( int i = 0; i < 2; i++ )
    {
        inputVars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( inputVars );
    neuronToMarabouMap[0] = inputVars;

    // Node 1 (fc1): 4 variables
    std::vector<Variable> fc1Vars;
    for ( int i = 0; i < 4; i++ )
    {
        fc1Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( fc1Vars );
    neuronToMarabouMap[1] = fc1Vars;

    // Node 2 (relu1): 4 variables
    std::vector<Variable> relu1Vars;
    for ( int i = 0; i < 4; i++ )
    {
        relu1Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( relu1Vars );
    neuronToMarabouMap[2] = relu1Vars;

    // Node 3 (fc2): 4 variables
    std::vector<Variable> fc2Vars;
    for ( int i = 0; i < 4; i++ )
    {
        fc2Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( fc2Vars );
    neuronToMarabouMap[3] = fc2Vars;

    // Node 4 (relu2): 4 variables
    std::vector<Variable> relu2Vars;
    for ( int i = 0; i < 4; i++ )
    {
        relu2Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( relu2Vars );
    neuronToMarabouMap[4] = relu2Vars;

    // Node 5 (fc3): 3 variables
    std::vector<Variable> fc3Vars;
    for ( int i = 0; i < 3; i++ )
    {
        fc3Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( fc3Vars );
    neuronToMarabouMap[5] = fc3Vars;

    // Node 6 (relu3): 3 variables
    std::vector<Variable> relu3Vars;
    for ( int i = 0; i < 3; i++ )
    {
        relu3Vars.push_back( Variable( varIndex++ ) );
    }
    marabouVars.push_back( relu3Vars );
    neuronToMarabouMap[6] = relu3Vars;

    // Node 7 (fc4): 1 variable
    std::vector<Variable> fc4Vars;
    fc4Vars.push_back( Variable( varIndex++ ) );
    marabouVars.push_back( fc4Vars );
    neuronToMarabouMap[7] = fc4Vars;

    // Set sizes for nodes
    inputNode->setInputSize( 2 );
    inputNode->setOutputSize( 2 );

    linearNode1->setInputSize( 2 );
    linearNode1->setOutputSize( 4 );

    reluNode1->setInputSize( 4 );
    reluNode1->setOutputSize( 4 );

    linearNode2->setInputSize( 4 );
    linearNode2->setOutputSize( 4 );

    reluNode2->setInputSize( 4 );
    reluNode2->setOutputSize( 4 );

    linearNode3->setInputSize( 4 );
    linearNode3->setOutputSize( 3 );

    reluNode3->setInputSize( 3 );
    reluNode3->setOutputSize( 3 );

    linearNode4->setInputSize( 3 );
    linearNode4->setOutputSize( 1 );

    // Convert std containers to Marabou containers
    Vector<std::shared_ptr<NLR::BoundedTorchNode>> marabouNodes;
    for ( const auto &node : nodes )
    {
        marabouNodes.append( node );
    }

    Vector<Vector<Variable>> marabouVarsVec;
    for ( const auto &vars : marabouVars )
    {
        Vector<Variable> marabouVarsInner;
        for ( const auto &var : vars )
        {
            marabouVarsInner.append( var );
        }
        marabouVarsVec.append( marabouVarsInner );
    }

    Vector<unsigned> marabouInputIndices;
    for ( const auto &idx : inputIndices )
    {
        marabouInputIndices.append( idx );
    }

    Map<unsigned, Vector<Variable>> marabouNeuronMap;
    for ( const auto &pair : neuronToMarabouMap )
    {
        Vector<Variable> marabouVarsInner;
        for ( const auto &var : pair.second )
        {
            marabouVarsInner.append( var );
        }
        marabouNeuronMap[pair.first] = marabouVarsInner;
    }

    Map<unsigned, Vector<unsigned>> marabouDependencies;
    for ( const auto &pair : dependencies )
    {
        Vector<unsigned> marabouDeps;
        for ( const auto &dep : pair.second )
        {
            marabouDeps.append( dep );
        }
        marabouDependencies[pair.first] = marabouDeps;
    }

    // Create and return the TorchModel
    NLR::TorchModel *model =
        new NLR::TorchModel( marabouNodes, marabouInputIndices, outputIndex, marabouDependencies );

    return model;
}

static void printBoundsTensor1D( const torch::Tensor &lower, const torch::Tensor &upper )
{
    torch::Tensor lb = lower;
    torch::Tensor ub = upper;
    if ( lb.dim() == 0 )
        lb = lb.unsqueeze( 0 );
    if ( ub.dim() == 0 )
        ub = ub.unsqueeze( 0 );
    for ( int i = 0; i < lb.size( 0 ); ++i )
    {
        if ( i > 0 )
            std::cout << " ";
        auto l = lb[i];
        auto u = ub[i];
        if ( l.dim() > 0 )
            l = l.flatten()[0];
        if ( u.dim() > 0 )
            u = u.flatten()[0];
        std::cout << "[" << l.item<double>() << ", " << u.item<double>() << "]";
    }
    std::cout << std::endl;
}

void runAlphaCROWNAnalysis( NLR::TorchModel *model, const std::string &networkName,
                            unsigned iterations = 20 )
{

    (void)networkName;

    try
    {
        // Prepare input bounds
        unsigned inputSize = model->getInputSize();
        torch::Tensor lowerBounds = torch::full( { (long)inputSize }, -1.0, torch::kFloat64 );
        torch::Tensor upperBounds = torch::ones( { (long)inputSize }, torch::kFloat64 );
        BoundedTensor<torch::Tensor> inputBounds( lowerBounds, upperBounds );

        // Configure analysis using LunaConfiguration
        LunaConfiguration::ANALYSIS_METHOD = LunaConfiguration::AnalysisMethod::AlphaCROWN;
        LunaConfiguration::ALPHA_ITERATIONS = iterations;
        LunaConfiguration::ALPHA_LR = 0.5f;
        LunaConfiguration::OPTIMIZE_LOWER = true;
        LunaConfiguration::OPTIMIZE_UPPER = true;
        LunaConfiguration::VERBOSE = true;

        // Run Alpha-CROWN analysis using unified compute_bounds() method
        BoundedTensor<torch::Tensor> alphaCrownResult =
            model->compute_bounds( inputBounds,
                                   nullptr, // No specification matrix
                                   LunaConfiguration::AnalysisMethod::AlphaCROWN,
                                   true, // compute lower bounds
                                   true  // compute upper bounds
            );

        torch::Tensor lowerBounds_result = alphaCrownResult.lower();
        torch::Tensor upperBounds_result = alphaCrownResult.upper();

        // Run clean CROWN baseline for comparison using new API
        std::cout << "\nRunning CROWN baseline for comparison..." << std::endl;

        // Reset all ReLU nodes to initialization mode for clean CROWN baseline
        for ( unsigned i = 0; i < model->getNodes().size(); ++i )
        {
            auto node = model->getNodes()[i];
            if ( node && node->getNodeType() == NLR::NodeType::RELU )
            {
                auto reluNode = std::dynamic_pointer_cast<NLR::BoundedReLUNode>( node );
                if ( reluNode )
                {
                    reluNode->setOptimizationStage( "init" );
                    reluNode->setAlphaCrownAnalysis( nullptr ); // Disable alpha optimization
                }
            }
        }

        // Run CROWN using compute_bounds() method
        BoundedTensor<torch::Tensor> crownResult =
            model->compute_bounds( inputBounds,
                                   nullptr, // No specification matrix
                                   LunaConfiguration::AnalysisMethod::CROWN,
                                   true, // compute lower bounds
                                   true  // compute upper bounds
            );

        std::cout << "\nCROWN:" << std::endl;
        if ( crownResult.lower().defined() && crownResult.upper().defined() )
        {
            printBoundsTensor1D( crownResult.lower(), crownResult.upper() );
        }
        else
        {
            std::cout << "[undefined]" << std::endl;
        }

        // Print final output bounds
        std::cout << "\nAlpha-CROWN:" << std::endl;
        if ( lowerBounds_result.defined() && upperBounds_result.defined() )
        {
            printBoundsTensor1D( lowerBounds_result, upperBounds_result );
        }
        else
        {
            std::cout << "[undefined]" << std::endl;
        }
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error in runAlphaCROWNAnalysis: " << e.what() << std::endl;
        throw;
    }
}

int main( int argc, char **argv )
{
    // Parse command line arguments for configuration
    LunaConfiguration::parseArgs( argc, argv );

    std::string networkType = "both"; // existing default
    unsigned iterations = LunaConfiguration::ALPHA_ITERATIONS;

    for ( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );
        if ( arg == "--deeppoly" )
        {
            networkType = "deeppoly";
        }
        else if ( arg == "--deeper" )
        {
            networkType = "deeper";
        }
        else if ( arg == "--single" )
        {
            networkType = "single";
        }
        else if ( arg == "--all" )
        {
            networkType = "all";
        }
        else if ( arg == "--iterations" && i + 1 < argc )
        {
            iterations = std::stoi( argv[i + 1] );
            i++;
        }
        else if ( arg == "--help" )
        {
            // print help if you want
            return 0;
        }
    }

    try
    {
        // DeepPoly (unchanged behavior for "deeppoly" or "both")
        if ( networkType == "deeppoly" || networkType == "both" || networkType == "all" )
        {
            NLR::TorchModel *deepPolyModel = createDeepPolyNetwork();
            // runAlphaCROWNAnalysis(deepPolyModel, "DeepPoly Network", iterations);
            delete deepPolyModel;
        }

        // Deeper
        if ( networkType == "deeper" || networkType == "both" || networkType == "all" )
        {
            NLR::TorchModel *deeperModel = createDeeperNetwork();
            runAlphaCROWNAnalysis( deeperModel, "Deeper Network", iterations );
            delete deeperModel;
        }

        // Single pathway
        if ( networkType == "single" || networkType == "all" )
        {
            NLR::TorchModel *singleModel = createSinglePathwayNetwork();
            // runAlphaCROWNAnalysis(singleModel, "SinglePathway Network", iterations);
            delete singleModel;
        }

        return 0;
    }
    catch ( const std::exception &e )
    {
        std::cerr << "Error in main: " << e.what() << std::endl;
        return 1;
    }
}
