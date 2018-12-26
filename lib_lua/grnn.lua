require 'nn'
require('./CGene.lua')
require('./CGeneDataTrim.lua')
require('./CMultiClamp.lua')
require('./CParamCache.lua')
local trainerPool = require('./grnnTrainerPool.lua')
local testerPool = require('../../MyCommon/testerPool.lua')

local grnn = {}

do
  function grnn.create(CModel, oDepGraph, taGERanges)
    local mNet = nn.Sequential()
    local taNonTF = oDepGraph:getNonTFs()
    local oParamCache = CParamCache.new()
    
    for __, strGene in pairs(taNonTF) do
      local mGene = CGene.new(CModel, strGene, oDepGraph, taGERanges, oParamCache)
      mNet:add(mGene)
    end
    
    mNet:add(CGeneDataTrim.new(oDepGraph))
    
    return mNet
  end

  function grnn.pri_trainTogether(mNet, taData)
    local taTrainParams = { nMaxIteration = 10, strOptimMethod = "SGD"}
    trainerPool.trainGrnnMNet(mNet, taData.input, taData.target, taTrainParams)
  end
  
  function grnn.train(mNet, taData)
    print("grnn.train, begin")
    local isFast = false
    local isReplaceTargetAfter = true
    
    local teTarget = taData.target:clone()
    local outputPrev = taData.input

      -- train each unit
      for i=1, #mNet.modules-1 do
      --for i=#mNet.modules-1, 1, -1 do
        local mUnit = mNet.modules[i]
        mUnit:initTrain(taData, isFast, isReplaceTargetAfter, outputPrev)
        outputPrev = mUnit:forward(outputPrev)
      end
      
    taData.target:copy(teTarget)
    
    print(testerPool.getMSE(mNet, taData.input, taData.target))
    
    --grnn.pri_trainTogether(mNet, taData)
    
    --print(testerPool.getMSE(mNet, taData.input, taData.target))
    
    print("grnn.train, end")
  end
  
  function grnn.predict(mNet, input)
    --mNet:add(nn.CMultiClamp(taMinMax.teMins, taMinMax.teMaxs))
    teOutput = mNet:forward(input)
    return teOutput
  end

  
  return grnn
end
