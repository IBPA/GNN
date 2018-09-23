-- CGeneDataTrim: extra data accumulated by sequence of CGene modules
CGeneDataTrim = torch.class('nn.CGeneDataTrim', 'nn.Module')

function CGeneDataTrim:__init(oDepGraph)
  self.nOutput = oDepGraph.nSize - oDepGraph.nTFs
  self.nFirstNonTFId = oDepGraph.nTFs + 1
end

function CGeneDataTrim:updateOutput(input)
  -- 1) KO (second item from input table is unnecessary
  -- 2) first nTF columns from first item from input table is unnecessary
  self.output = input[1]:narrow(2, self.nFirstNonTFId, self.nOutput)
  return self.output
end

function CGeneDataTrim:pri_ensureGradInput(teInput)
  self.gradInput = self.gradInput or torch.Tensor()
  self.gradInput:resizeAs(teInput)
end

function CGeneDataTrim:updateGradInput(input, gradOutput) 
  -- notes: 1) no need to include KO info in the grad input, 2) gradInput for TFs is allways zero
  self:pri_ensureGradInput(input[1])
  self.gradInput:fill(0)
  local gradInputSlice = self.gradInput:narrow(2, self.nFirstNonTFId, self.nOutput)
  gradInputSlice:copy(gradOutput)
  
  return self.gradInput
end
