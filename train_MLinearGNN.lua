--[[ Description:  Build and train GNN model 
    Arg1: base directory name (containing net.dep, train_MR.tsv, train_NMR.tsv and train_KO.tsv).
    The output trained_GNN.model will be also saved Arg1 directory.
]]

require('./lib_lua/CMLinear.lua')
require('./lib_lua/graph.lua')
require('./lib_lua/data.lua')
local grnn = require('./lib_lua/grnn.lua')

function fuGetFilenames(strDir)
    local strFilenameTF = string.format("%s/train_MR.tsv", strDir)
    local strFilenameNonTF = string.format("%s/train_NMR.tsv", strDir)
    local strFilenameKO = string.format("%s/train_KO.tsv", strDir)
    local strFilenameStratifiedIds = nil

    return strFilenameTF, strFilenameNonTF, strFilenameKO, strFilenameStratifiedIds
end

torch.manualSeed(0)
local strDir = arg[1]

-- 0) Load Data
local oDepGraph = CDepGraph.new(string.format("%s/net.dep", strDir))
local mDataTrain = CData.new(strDir, oDepGraph, nil, false, nil, nil, fuGetFilenames)

-- 1) Build Model
local mNet = grnn.create(CMLinear, oDepGraph, mDataTrain.taGERanges)

-- 2) Train Model
grnn.train(mNet, mDataTrain.taData)

-- 3) Save model
local strModelFilename = string.format("%s/trained_MLinearGNN.model", strDir)
torch.save(strModelFilename, mNet)
print(string.format("Model saved to: '%s'", strModelFilename))
