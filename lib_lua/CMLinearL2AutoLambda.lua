--[[ Description: CLinear class implements linear GNN module based on CMLinear but using L2 regularization with auto lambda determined by k-fold cross validation method.
]]
require 'math'
local dataLoad = dataLoad or require('./common_dataLoad.lua')
CMLinear = require('../lib_lua/CMLinear.lua')

CMLinearL2AutoLambda = torch.class('CMLinearL2AutoLambda', 'CMLinear')

-- get mean squared error
function CMLinearL2AutoLambda:getMSE(teX, teTheta, teY)
	local mSE = (teY - torch.mm(teX, teTheta)):pow(2):mean()
	return mSE
end

-- get beta, which is teTheta in the training process
function CMLinearL2AutoLambda:getBeta(dLambda, teInput, teTarget)
  local teInputWB = teInput -- input with bias
  local teInputWBT = teInputWB:transpose(1, 2) -- input with bias transpose matrix
  local teOutput = teTarget
  local lambda = dLambda

  local teBeta = torch.mm(torch.mm(torch.inverse(torch.mm(teInputWBT, teInputWB) + torch.diag(lambda * torch.ones(teInputWBT:size(1)))), teInputWBT), teOutput)
  return teBeta
end

-- create a mask for k-fold cross validation
function CMLinearL2AutoLambda:maskForKFold(teInput, teTarget, foldID, nFolds)
  local nSize = teInput:size(1)
  local teIdx = torch.linspace(1, nSize, nSize)

  -- train:
  local teTrainMask = torch.mod(teIdx, nFolds):ne(torch.Tensor(nSize):fill(foldID))
  local teTrain_input = dataLoad.pri_getMasked(teInput, teTrainMask)
  local teTrain_target = dataLoad.pri_getMasked(teTarget, teTrainMask)

  -- test
  local teTestMask = torch.mod(teIdx, nFolds):eq(torch.Tensor(nSize):fill(foldID))
  local teTest_input = dataLoad.pri_getMasked(teInput, teTestMask)
  local teTest_target = dataLoad.pri_getMasked(teTarget, teTestMask)

  return teTrain_input, teTrain_target, teTest_input, teTest_target
end

-- get cross validation error
function CMLinearL2AutoLambda:getCVErr(l, teInput, teTarget, nFolds)
	local nFoldModMax = nFolds - 1
	local dTotalError = 0
  local nCountFold = 0

	for foldID = 0, nFoldModMax, 1 do
		teTrain_input, teTrain_target, teTest_input, teTest_target = self:maskForKFold(teInput, teTarget, foldID, nFolds)
    if ((teTrain_input ~= nil) and (teTrain_target ~= nil) and (teTest_input  ~= nil) and (teTest_target ~= nil)) then
		  teBeta = self:getBeta(l, teTrain_input, teTrain_target)
      dTempErr = self:getMSE(teTest_input, teBeta, teTest_target)
		  dTotalError = dTotalError + dTempErr
      nCountFold = nCountFold + 1
    else
      print("Error in getCVErr call!!")
    end
	end

	dTotalError = dTotalError / nCountFold
	return dTotalError
end

function CMLinearL2AutoLambda:train(teInput, teTarget)
  local teInputExtended = self:pri_extendWithMulTerms(teInput)
  local teA = torch.cat(torch.ones(teInput:size(1), 1), teInputExtended)
	local dBestErr = math.huge
	local dBestLambda = 0
  local nTempKFold = 5
	-- k-fold cross validation on lambda(0, 2, 10)
  local maxPossibleK = teInput:size(1)
  if maxPossibleK < 5 then
    nTempKFold = maxPossibleK
  end

	for l = 0, 2, 0.2 do
		tempErr = self:getCVErr(l, teA, teTarget, nTempKFold)
		if tempErr < dBestErr then
			dBestErr = tempErr
			dBestLambda = l
		end
	end

	local teBestTheta = self:getBeta(dBestLambda, teA, teTarget)
	self.teTheta:copy(teBestTheta)
end
