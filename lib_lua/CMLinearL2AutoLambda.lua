--[[ Description: CLinear class implements linear GNN module based on CMLinear but using L2 regularization with auto lambda determined by k-fold cross validation method.
]]
require 'math'
local dataLoad = dataLoad or require('./common_dataLoad.lua')

CMLinearL2AutoLambda = torch.class('CMLinearL2AutoLambda')

function CMLinearL2AutoLambda:__init(nInputs, taMins, taMaxs)
  self.taMins = taMins
  self.taMaxs = taMaxs

  self.nInputs = nInputs
  self.nMulTerms = (nInputs * (nInputs-1))/2
  self.teTheta = torch.zeros(1, self.nInputs + self.nMulTerms + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
end

function CMLinearL2AutoLambda:pri_extendWithMulTerms(teInput)
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

function CMLinearL2AutoLambda:predict(teInput)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local teV1 = torch.Tensor(teInputExtended:size(1), 1):fill(self.teTheta[1][1]) -- bias
  local teV2 = self.teTheta:narrow(2, 2, self.teTheta:size(2)-1):t()
  local teOutput = torch.addmm(teV1, teInputExtended, teV2)
  teOutput:clamp(self.taMins.output, self.taMaxs.output)

  return teOutput
end

function CMLinearL2AutoLambda:getMSE(teX, teTheta, teY)
	local mSE = (teY - torch.mm(teX, teTheta)):pow(2):mean()
	return mSE
end



function CMLinearL2AutoLambda:getBeta(dLambda, teInput, teTarget)

  local Z = teInput
  local ZT = Z:transpose(1, 2)
  local y = teTarget
  local lambda = dLambda

  local beta = torch.mm(torch.mm(torch.inverse(torch.mm(ZT, Z) + torch.diag(lambda * torch.ones(ZT:size(1)))), ZT), y)
  return beta
end


function CMLinearL2AutoLambda:maskForKFold(teInput, teTarget, foldID, nFolds)
  local nSize = teInput:size(1)
  local teIdx = torch.linspace(1, nSize, nSize)

  -- train:
  local trainMask = torch.mod(teIdx, nFolds):ne(torch.Tensor(nSize):fill(foldID))
  local teTrain_input = dataLoad.pri_getMasked(teInput, trainMask)
  local teTrain_target = dataLoad.pri_getMasked(teTarget, trainMask)


  -- test
  local testMask = torch.mod(teIdx, nFolds):eq(torch.Tensor(nSize):fill(foldID))
  local teTest_input = dataLoad.pri_getMasked(teInput, testMask)
  local teTest_target = dataLoad.pri_getMasked(teTarget, testMask)

  return teTrain_input, teTrain_target, teTest_input, teTest_target
end

function CMLinearL2AutoLambda:getCVErr(l, teInput, teTarget, kfold)
	local nFoldModMax = kfold - 1
	local totalError = 0
  local countFold = 0
	for foldID = 0, nFoldModMax, 1 do
		teTrain_input, teTrain_target, teTest_input, teTest_target = self:maskForKFold(teInput, teTarget, foldID, kfold)
    if ((teTrain_input ~= nil) and (teTrain_target ~= nil) and (teTest_input  ~= nil) and (teTest_target ~= nil)) then

		  teBeta = self:getBeta(l, teTrain_input, teTrain_target)
      tempErr = self:getMSE(teTest_input, teBeta, teTest_target)
		  totalError = totalError + tempErr
      countFold = countFold + 1
    end
	end
	totalError = totalError / countFold
	return totalError
end


function CMLinearL2AutoLambda:train(teInput, teTarget)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local teA = torch.cat(torch.ones(teInput:size(1), 1), teInputExtended)
	local bestErr = math.huge
	local bestL = 0
  local tempKFold = 5
	-- k-fold cross validation on lambda(0, 2, 10)
  local maxPossibleK = teInput:size(1)
  if maxPossibleK < 5 then
    tempKFold = maxPossibleK
  end

	for l = 0, 2, 0.2 do
		tempErr = self:getCVErr(l, teA, teTarget, tempKFold)
		if tempErr < bestErr then
			bestErr = tempErr
			bestL = l
		end
	end

	local teBestTheta = self:getBeta(bestL, teA, teTarget)
	self.teTheta:copy(teBestTheta)

end

function CMLinearL2AutoLambda:getParamPointer()
  return self.teTheta
end

function CMLinearL2AutoLambda:getGradParamPointer()
  return self.teGradTheta
end
