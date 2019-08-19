--[[ Description: CLinear class implements linear GNN module with multiplicative terms.
]]
require 'math'
require('./common_dataLoad.lua')

CMLinear = torch.class('CMLinear')

function CMLinear:__init(nInputs, taMins, taMaxs)
  self.taMins = taMins
  self.taMaxs = taMaxs

  self.nInputs = nInputs
  self.nMulTerms = (nInputs * (nInputs-1))/2
  self.teTheta = torch.zeros(1, self.nInputs + self.nMulTerms + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
end

function CMLinear:pri_extendWithMulTerms(teInput)
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

function CMLinear:predict(teInput)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local teV1 = torch.Tensor(teInputExtended:size(1), 1):fill(self.teTheta[1][1]) -- bias
  local teV2 = self.teTheta:narrow(2, 2, self.teTheta:size(2)-1):t()
  local teOutput = torch.addmm(teV1, teInputExtended, teV2)
  teOutput:clamp(self.taMins.output, self.taMaxs.output)

  return teOutput
end

function getMSE(teX, teTheta, teY)
	local mSE = (teY - torch.mm(teX, teTheta)):pow(2):mean()
	return mSE
end

function getBeta(l, teInput, teTarget)
  local teInputExtended = self:pri_extendWithMulTerms(teInput) 
  local Z = torch.cat(torch.ones(teInputExtended:size(1), 1), teInputExtended)
  local ZT = Z:transpose(1, 2)
  local y = teTarget
  local lambda = l
  -- local ZTZ = torch.mm(ZT, Z)
  -- local lambdaIp = torch.diag(lambda * torch.ones(ZT:size(1)))
  -- local ZTZIPinv = torch.inverse(torch.mm(ZT, Z) + torch.diag(lambda * torch.ones(ZT:size(1))))
  -- local ZTY = torch.mm(torch.inverse(torch.mm(ZT, Z) + torch.diag(lambda * torch.ones(ZT:size(1)))), ZT)
  local beta = torch.mm(torch.mm(torch.inverse(torch.mm(ZT, Z) + torch.diag(lambda * torch.ones(ZT:size(1)))), ZT), y)
  return beta
end

function getCVErr(l, teX, teTarget, kfold)
	local nFoldModMax = kfold - 1
	local totalError = 0
	for foldID = 0, nFoldModMax do
		taTrain, taTest = dataload.loadTrainTestForCV(taParam, foldID)
		teBeta = getBeta(l, taTrain[1], teTarget[2])
		totalError = totalError + getMSE(taTest[1], teBeta, taTest[2])
	end
	totalError = totalError / kfold
	return totalError
end


function CMLinear:train(teInput, teTarget)

	local bestErr = math.huge
	local bestL = 0

  	-- k-fold cross validation on lambda(0, 2, 10)
  	local tempKFold = 5
	for l = 0, 2, 10 do
		tempErr = getCVErr(l, teX, teTarget, tempKFold)
		if tempErr < bestErr then
			bestErr = tempErr
			bestL = l
		end
		-- update if needed
	end
	local teBestTheta = getBeta(bestL, teX, teTarget)
	self.teTheta:copy(teBestTheta)

end

function CMLinear:getParamPointer()
  return self.teTheta
end

function CMLinear:getGradParamPointer()
  return self.teGradTheta
end
