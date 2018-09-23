CLinear = torch.class('CLinear')

function CLinear:__init(nInputs)
  self.nInputs = nInputs
  self.teTheta = torch.zeros(1, nInputs + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
end

function CLinear:predict(teInput)
  local teV1 = torch.Tensor(teInput:size(1), 1):fill(self.teTheta[1][1]) -- bias
  local teV2 = self.teTheta:narrow(2, 2, self.teTheta:size(2)-1):t()
  local teOutput = torch.addmm(teV1, teInput, teV2)
  return teOutput
end

function CLinear:train(teInput, teTarget)
  local teA = torch.cat(torch.ones(teInput:size(1), 1), teInput)
  local teB = teTarget
  local teX = torch.gels(teA, teB)
  self.teTheta:copy(teX)
end

function CLinear:getGradInput(teInput, teGradOutput)
  local teV2 = self.teTheta:narrow(2, 2, self.teTheta:size(2)-1)
  return torch.mm(teGradOutput, teV2)
end

function CLinear:getParamPointer()
  return self.teTheta
end

function CLinear:getGradParamPointer()
  return self.teGradTheta
end

function  CLinear:accGradParameters(teInput, teGradOutput, scale)
  local teGrad = self.teGradTheta:narrow(2, 2, self.teGradTheta:size(2)-1)
  teGrad:addmm(scale, teGradOutput:t(), teInput)
  
  self.teGradTheta[1][1] = self.teGradTheta[1][1] + torch.sum(teGradOutput)*scale
end

