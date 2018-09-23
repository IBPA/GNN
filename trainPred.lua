-- Given dataset directory, train and predict.
require('./lib_lua/graph.lua')

require('./lib_lua/data.lua')
local grnn = require('./lib_lua/grnn.lua')
local debug_model = require('./lib_lua/debug_model.lua')

torch.manualSeed(0)

local strDir = arg[1] or "/Users/ameen/mygithub/grnnPipeline/data/dream5_med2/modules_gnw_a/size-10/top_edges-1_gnw_data/d_1/"
local strFoldFilename = arg[2] or "n10_f1.txt"
local isNoise = true

-- 0) Load Data
local oDepGraph = CDepGraph.new(string.format("%s/../net.dep", strDir))
local mDataTrain = CData.new(strDir, oDepGraph, nil, isNoise, strFoldFilename, 1)

-- 1) Build Model
local mNet = grnn.create(oDepGraph, mDataTrain.taGERanges)

-- 2) Train ToDo: skip k_Fold for now, but do it with an eye on available functionality
grnn.train(mNet, mDataTrain.taData)
local taMinMax = mDataTrain:getMinMaxNonTFs()

print("test")
local mDataTest = CData.new(strDir, oDepGraph, nil, isNoise, strFoldFilename, 2)
local teOutput = grnn.predict(mNet, mDataTest.taData.input, taMinMax)

--print("save debug info")
local strDebugDir = string.format("%s/grnn_debug/%s/", strDir, strFoldFilename)
debug_model.saveModelDebug(mNet, mDataTrain, mDataTest, oDepGraph, strDebugDir)

mDataTest:savePred(teOutput, "grnn_")
mDataTest:saveActual()

