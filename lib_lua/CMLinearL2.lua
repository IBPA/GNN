--[[ Description: CLinear class implements linear GNN module with multiplicative terms.
]]

CMLinearL2 = torch.class('CMLinearL2')

function CMLinearL2:__init(nInputs, taMins, taMaxs, taLambda)
  self.taMins = taMins
  self.taMaxs = taMaxs
  self.talambda = taLambda

  self.nInputs = nInputs
  self.nMulTerms = (nInputs * (nInputs-1))/2
  self.teTheta = torch.zeros(1, self.nInputs + self.nMulTerms + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
end

function CMLinearL2:pri_extendWithMulTerms(teInput)
  local nD = teInput:size(2)
  if nD < 2 then
    return teInput
  end

  -- 1) Build matrix of for all pairs of input multiplies
  local teMulTerms = torch.zeros(teInput:size(1), self.nMulTerms)
  local idMul = 0
  for i=1,nD do
    for j=i+1,nD do
      idMul = idMul + 1
      teMulTerms:select(2, idMul):copy(torch.cmul(teInput:select(2, i), teInput:select(2, j)))
    end
  end

  -- 2) Extend
  return torch.cat({teInput, teMulTerms}, 2)
end

function CMLinearL2:predict(teInput)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local teV1 = torch.Tensor(teInputExtended:size(1), 1):fill(self.teTheta[1][1]) -- bias
  local teV2 = self.teTheta:narrow(2, 2, self.teTheta:size(2)-1):t()
  local teOutput = torch.addmm(teV1, teInputExtended, teV2)
  teOutput:clamp(self.taMins.output, self.taMaxs.output)

  return teOutput
end

function CMLinearL2:train(teInput, teTarget)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local Z = torch.cat(torch.ones(teInputExtended:size(1), 1), teInputExtended)
  local ZT = Z:transpose(1, 2)
  local y = teTarget


  print(Z)
  print("---z---")
  print(ZT)
  print("---zt---")
  print(y)
  print("---y---")
  local teX
  function fuWrapGels()
    teX = torch.mm(torch.mm(torch.inverse(torch.mm(ZT, Z) + torch.diag(self.talambda * torch.ones(ZT:size(1)))), ZT), y)
  end

  if pcall(fuWrapGels) then
     self.teTheta:copy(teX)
  else
     print("Error in gels call!!")
     self.teTheta:fill(0)
     self.teTheta[1][1] = torch.mean(teTarget)
  end
end

function CMLinearL2:getParamPointer()
  return self.teTheta
end

function CMLinearL2:getGradParamPointer()
  return self.teGradTheta
end
