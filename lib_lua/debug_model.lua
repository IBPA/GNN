local myUtil = myUtil or require('../../MyCommon/util.lua')
local debug_model = {}

do
  function getInputDebug(nDebugPoints, oDepGraph, mDataTrain)
      local nInputs = mDataTrain.taData.input[1]:size(2)
      local teInput = torch.rand(nDebugPoints, nInputs)
      local taTFOrders = oDepGraph:getTFOrders()
      
      for strTF, nOrder in pairs(taTFOrders) do
        local taRange = mDataTrain.taGERanges[strTF]
        teInput:select(2, nOrder):mul(taRange.max - taRange.min)
        teInput:select(2, nOrder):add(taRange.min)
      end
      
      local teKO = torch.ones(nDebugPoints, mDataTrain.taData.input[2]:size(2))
      
      return {teInput, teKO}
      
  end
  
  function getOrdered(teInput, taOrder)
    local teOutput = teInput:clone()
    local taHeader = {}
    
    local idx = 1
    for strGene, nOrder in pairs(taOrder) do
      table.insert(taHeader, strGene)
      teOutput:select(2, idx):copy(teInput:select(2, nOrder))
      idx = idx + 1
    end
    
    return teOutput, taHeader
  end
  
  
  function  saveDebug(teInputDebug, teOutputDebug, oDepGraph, mDataTrain, strDebugDir)
      os.execute("mkdir -p " .. strDebugDir)
      local strNonTFsPredFilename = string.format("%s/NonTFs.csv", strDebugDir)
      local teNonTFPredReordered = data.priGetDataOrigOrder(teOutputDebug, mDataTrain.taNonTFGenes, oDepGraph:getNonTFOrders())
      myUtil.saveTensorAndHeaderToCsvFile(teNonTFPredReordered, mDataTrain.taNonTFGenes, strNonTFsPredFilename)
      print("saved debug preds to:" .. strNonTFsPredFilename)
      
      
      local strTFsPredFilename = string.format("%s/TFs.csv", strDebugDir)
      local teInputDebugOrdered, taTFHeader = getOrdered(teInputDebug, oDepGraph:getTFOrders())
      myUtil.saveTensorAndHeaderToCsvFile(teInputDebugOrdered, taTFHeader, strTFsPredFilename)
      print("saved debug input to:" .. strNonTFsPredFilename)
      
    end
  function debug_model.debugInfoToString(taDebugInfo)
    local strTrainError = taDebugInfo.dTrainErr and string.format("%.7f", taDebugInfo.dTrainErr) or "nan"
    return string.format("%s,%s,%s,%.7f,%s",
      taDebugInfo.strGene,
      myUtil.getCsvStringFrom1dTensor(taDebugInfo.teP), 
      myUtil.getCsvStringFrom1dTensor(taDebugInfo.teInputMean), 
      taDebugInfo.teTargetMean,
      strTrainError)
  end
  
  function debug_model.saveParamDebugInfo(mNet, strDebugDir)
      local strLog = ""
      for i=1, #mNet.modules-1 do
      --for i=#mNet.modules-1, 1, -1 do
        local mUnit = mNet.modules[i]
        strLog = strLog .. debug_model.debugInfoToString(mUnit:getDebugInfo()) .. "\n"
      end
      
      os.execute("mkdir -p " .. strDebugDir)
      local filename = string.format("%s/paramDebug.csv", strDebugDir)
      local f = io.open(filename, "w")
      f:write(strLog)
      f:close()
      print("Logged: " .. filename)
  end
  
  function debug_model.saveModelDebug(mNet, mDataTrain, mDataTest, oDepGraph, strDebugDir)
    --local nDebugPoints = 100
    --local taInputDebug = getInputDebug(nDebugPoints, oDepGraph, mDataTrain)
    --teOutputDebug = mNet:forward(taInputDebug)
    --saveDebug(taInputDebug[1], teOutputDebug, oDepGraph, mDataTrain, strDebugDir)
    
    debug_model.saveParamDebugInfo(mNet, strDebugDir)

  end

return debug_model
end