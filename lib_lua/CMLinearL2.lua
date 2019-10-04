--[[ Description: CLinear class implements linear GNN module based on CMLinear but using L2 regularization with lambda as parameter.
]]
CMLinear = require('../lib_lua/CMLinear.lua')
CMLinearL2 = torch.class('CMLinearL2', 'CMLinear')

function CMLinearL2:__init(nInputs, taMins, taMaxs, dLambda)
  self.taMins = taMins
  self.taMaxs = taMaxs
  self.dlambda = dLambda

  self.nInputs = nInputs
  self.nMulTerms = (nInputs * (nInputs-1))/2
  self.teTheta = torch.zeros(1, self.nInputs + self.nMulTerms + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
end

function CMLinearL2:train(teInput, teTarget)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  -- add bias vectore to the extended input tensor matrix
  local teInputWB = torch.cat(torch.ones(teInputExtended:size(1), 1), teInputExtended) -- input with bias
  local teInputWBT = teInputWB:transpose(1, 2) -- input with bias transpose matrix
  local teOutput = teTarget

  local teBeta -- local variable storing teTheta
  -- function that calculate beta
  function fuCalcBeta()
    teBeta = torch.mm(torch.mm(torch.inverse(torch.mm(teInputWBT, teInputWB) + torch.diag(self.dlambda * torch.ones(teInputWBT:size(1)))), teInputWBT), teOutput)
  end

  if pcall(fuCalcBeta) then
     self.teTheta:copy(teBeta)
  else
     print("Error in gels call!!")
     self.teTheta:fill(0)
     self.teTheta[1][1] = torch.mean(teTarget)
  end
end
