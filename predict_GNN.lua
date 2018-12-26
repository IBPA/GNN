-- Load GNN model and predict

require('./lib_lua/CSyng.lua')
require('./lib_lua/graph.lua')
require('./lib_lua/data.lua')
local grnn = require('./lib_lua/grnn.lua')
local myUtil = myUtil or require('../../MyCommon/util.lua')

function fuGetFilenames(strDir)
    local strFilenameTF = string.format("%s/input_MR.tsv", strDir)
    local strFilenameNonTF = nil
    local strFilenameKO = string.format("%s/input_KO.tsv", strDir)
    local strFilenameStratifiedIds = nil

    return strFilenameTF, strFilenameNonTF, strFilenameKO, strFilenameStratifiedIds
end

torch.manualSeed(0)
local strDir = arg[1]
local strOutputFilename = arg[2]

-- 0) Load Data
local oDepGraph = CDepGraph.new(string.format("%s/net.dep", strDir))
local mData = CData.new(strDir, oDepGraph, nil, false, nil, nil, fuGetFilenames)

-- 1) Load Model
local strModelFilename = string.format("%s/trained_GNN.model", strDir)
mNet = torch.load(strModelFilename)

-- 2) Predict
local teOutput = mNet:forward(mData.taData.input)

-- 3) Save
local taGeneNames = oDepGraph:getNonTFs()
myUtil.saveTensorAndHeaderToCsvFile(teOutput, taGeneNames,  strOutputFilename)
print(string.format("saved preds to '%s'.", strOutputFilename))
