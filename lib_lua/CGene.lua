require('./CLinear.lua')
require('./CSyng.lua')
CGene = torch.class('nn.CGene', 'nn.Module')

function getGeneAndTFRanges(taGERanges, strGene, taInputNames)
  local taMins = { output = 0, inputs = torch.zeros(#taInputNames) }
  local taMaxs = { output = 1, inputs = torch.ones(#taInputNames) }
  
  if taGERanges ~= nil then
    taMins.output = taGERanges[strGene].min
    taMaxs.output = taGERanges[strGene].max
    
    for i=1, #taInputNames do
      taMins.inputs[i] = taGERanges[taInputNames[i]].min
      taMaxs.inputs[i] = taGERanges[taInputNames[i]].max
    end
  end

  return taMins, taMaxs
end

function CGene:__init(strGene, oDepGraph, taGERanges, oParamCache)
  self.strGene = strGene
  self.oDepGraph = oDepGraph
  
  self.nInput = oDepGraph:getnBefore(self.strGene)
  self.nOutput = self.nInput + 1
  self.nNonTFId = self.nInput - oDepGraph.nTFs + 1
  
  local taInputNames
  self.taInputIdx, taInputNames = self.oDepGraph:getDepIds(self.strGene)
  self.taMins, self.taMaxs = getGeneAndTFRanges(taGERanges, strGene, taInputNames)
  self.dBasal = 0
  
--  self.model = CLinear.new(#self.taInputIdx)  -- using linear model for now
  self.model = CSyng.new(#self.taInputIdx, self.taMins, self.taMaxs, oParamCache)
  self.weight = self.model:getParamPointer()
  self.gradWeight = self.model:getGradParamPointer()
  
  self.taDebug = {strGene = self.strGene}
end

function CGene:pri_ensureOutput(input)
  self.output = {torch.cat(input[1], torch.zeros(input[1]:size(1), 1)),
                 input[2]}
    
end

function CGene:pri_tmpTableToCsvString(taInput)
  local strR = ""
  for k, v in pairs(self.taInputIdx) do
    strR = strR .. "," .. v
  end
  
  return strR
end


function CGene:pri_getInputSlice(teInput) -- Create a slice including only the dependant input columns
  --io.write("TFs: " .. self:pri_tmpTableToCsvString(self.taInputIdx) .. "\n")
    
  local teInputSlice = torch.Tensor(teInput:size(1), #self.taInputIdx)
  for i, depId in pairs(self.taInputIdx) do
    teInputSlice:narrow(2, i, 1):copy(teInput:narrow(2, depId, 1))
  end
  
  return teInputSlice
end

function CGene:pri_getKOSlice(teKO) -- return single slice with 1 where it's knockout and 0 elsewhere
  local teKOSlice = teKO:narrow(2, self.nNonTFId, 1)
  return teKOSlice:eq(torch.zeros(teKOSlice:size()))
end

function CGene:pri_updateInputDebugInfo(teInputSlice)
  self.taDebug.teInputMean = torch.mean(teInputSlice, 1):select(1, 1)
end


function CGene:pri_updateOutputSliceCalc(teInput, teOutputSlice)
  local teInputSlice = self:pri_getInputSlice(teInput)
  teOutputSlice:copy(self.model:predict(teInputSlice))
end

function CGene:pri_updateOutputSliceKO(teKO, teOutputSlice)
  local teKOSlice = self:pri_getKOSlice(teKO)
  teOutputSlice:maskedFill(teKOSlice, self.dBasal)
end

function CGene:updateOutput(input)
  --[[
  io.write("fw(" .. self.strGene .. ")")
  if self.strGene == "csgD" then
    print("just for debugging")
  end--]]
  
  self:pri_ensureOutput(input)
  local teOutputSlice = self.output[1]:narrow(2, self.output[1]:size(2), 1)
  self:pri_updateOutputSliceCalc(input[1], teOutputSlice)
  self:pri_updateOutputSliceKO(input[2], teOutputSlice)
  
  return self.output
end

function CGene:pri_updateGradInputSlice(gradInput, teLocalGradInputSlice)
  for i, depId in pairs(self.taInputIdx) do
    gradInput:narrow(2, depId, 1):copy(teLocalGradInputSlice:narrow(2, i, 1))
  end
end

function CGene:pri_ensureGradInput(teInput)
  self.gradInput = self.gradInput or torch.Tensor()
  self.gradInput:resizeAs(teInput)
end

function CGene:updateGradInput(input, gradOutput)
  -- 1) create gradInput, with same size of input[1]
  self:pri_ensureGradInput(input[1])
  
  -- 2) copy gradOutput into gradInput (except for  this gene's output. ( TFs, don't care about TF's gradInput, can ignore them.)
  self.gradInput:copy(gradOutput:narrow(2, 1, gradOutput:size(2) - 1))
  -- 3) Calculate gradInput given (a) gradOutput's last column (for this gene), (b) dependencies of this gene.
  -- Note; need to "add" calculated to existing values coming from gradOutput.
  local teInputSlice = self:pri_getInputSlice(input[1])
  local teGradOutputSlice = gradOutput:narrow(2, self.gradInput:size(2), 1)
  local teLocalGradInputSlice = self.model:getGradInput(teInputSlice, teGradOutputSlice)
  self:pri_updateGradInputSlice(self.gradInput, teLocalGradInputSlice)
  
  return self.gradInput
end

function CGene:pri_getInputReal(taData)
  local nRows = taData.input[1]:size(1)
  local teR = torch.Tensor(nRows, #self.taInputIdx)
  
  for i, depId in pairs(self.taInputIdx) do
    if depId <= self.oDepGraph.nTFs then -- look for it in the input
      teR:narrow(2, i, 1):copy(taData.input[1]:narrow(2, depId, 1))
    else -- look for it in the target
      local nTargetId = depId - self.oDepGraph.nTFs
      teR:narrow(2, i, 1):copy(taData.target:narrow(2, nTargetId, 1))
    end
  end 
  
  return teR
end

function CGene:pri_getTargetReal(teTarget)
  return teTarget:narrow(2, self.nNonTFId, 1):clone()
end

function CGene:pri_getMasked(teInput, teMask)
  --local nRowsMasked = teMask:select(2, 1):sum()
  local teIdx = teMask:nonzero()
  local teR = teInput:gather(1, teIdx:narrow(2, 1, 1):expand(teIdx:size(1), teInput:size(2)))
  return teR
end

function CGene:accGradParameters(input, gradOutput, scale)
    local teInputSlice = self:pri_getInputSlice(input[1])
    local teGradOutputSlice = gradOutput:narrow(2, input[1]:size(2), 1)
    self.model:accGradParameters(teInputSlice, teGradOutputSlice, scale)
end

function CGene:pri_getBasal(teKOSlice, teInputReal)
  if teKOSlice:sum() == 0 then -- if no KO if no KO info, then will pick zero.
    return self.taMins.output
  else    
    return torch.mean(teInputReal:maskedSelect(teKOSlice))
  end
end

function CGene:initTrain(taData, isFast, isReplaceTargetAfter, outputPrev)
  local teInputReal = self:pri_getInputReal(taData)
  local teTargetReal = self:pri_getTargetReal(taData.target)
  local teKOSlice = self:pri_getKOSlice(taData.input[2])
  
  self.dBasal = self:pri_getBasal(teKOSlice, teTargetReal)
  teInputReal = self:pri_getMasked(teInputReal, teKOSlice:ne(1))
  teTargetReal = self:pri_getMasked(teTargetReal, teKOSlice:ne(1))
    
  -- ToDo: crate new tensors for "non-KO" rows
  io.write("train " .. self.strGene .. ":")
  io.flush()
  self.taDebug.teP, self.taDebug.dTrainErr = self.model:train(teInputReal, teTargetReal, isFast)
  self:pri_updateInputDebugInfo(teInputReal)
  self.taDebug.teTargetMean = torch.mean(teTargetReal)
  io.write("\n")
  
  if isReplaceTargetAfter then
    local tePred = self:updateOutput(outputPrev)[1]
    local tePredSlice = tePred:narrow(2, tePred:size(2), 1)
    local teTargetSlice = taData.target:narrow(2, self.nNonTFId, 1)
    teTargetSlice:copy(tePredSlice)
  end
  
end

function CGene:getDebugInfo()
  return self.taDebug
  
end
