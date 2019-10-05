--[[ Description: CLinear class implements linear GNN module based on CMLinear but using L2 regularization with auto lambda determined by k-fold cross validation method.
]]
require 'math'
local dataLoad = dataLoad or require('./common_dataLoad.lua')
require('../lib_lua/CMLinearL2.lua')

CMLinearL2AutoLambda = torch.class('CMLinearL2AutoLambda', 'CMLinearL2')

function CMLinearL2AutoLambda:__init(nInputs, taMins, taMaxs)
  self.taMins = taMins
  self.taMaxs = taMaxs
  self.dlambda = dLambda

  self.nInputs = nInputs
  self.nMulTerms = (nInputs * (nInputs-1))/2
  self.teTheta = torch.zeros(1, self.nInputs + self.nMulTerms + 1)
  self.teGradTheta = torch.zeros(self.teTheta:size())
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

-- cross validation
function CMLinearL2AutoLambda:CrossValidation(oModule, teInput, teTarget, nFolds)
  local nFoldModMax = nFolds - 1
  local dTotalError = 0
  local nCountFold = 0

  for foldID = 0, nFoldModMax do
    local teTrain_input, teTrain_target, teTest_input, teTest_target = self:maskForKFold(teInput, teTarget, foldID, nFolds)
    if ((teTrain_input ~= nil) and (teTrain_target ~= nil) and (teTest_input  ~= nil) and (teTest_target ~= nil)) then
      oModule:train(teTrain_input, teTrain_target)
      dTempErr = (teTest_target - oModule:predict(teTest_input)):pow(2):mean()
      dTotalError = dTotalError + dTempErr
      nCountFold = nCountFold + 1
    else
      print("Error in CrossValidation!!")
    end
  end
  local meanErr = dTotalError / nCountFold
  return meanErr
end

function CMLinearL2AutoLambda:train(teInput, teTarget)
  local dBestErr = math.huge
  local dBestLambda = 0
  local nKFold = 5
  local MAX_LAMBDA = 2
  local LAMBDA_STEP = 0.2

  -- validate k-fold
  local maxPossibleK = teInput:size(1)
  if maxPossibleK < 5 then
    nKFold = maxPossibleK
  end

  for l = 0, MAX_LAMBDA, LAMBDA_STEP do
    local oM = CMLinearL2.new(self.nInputs, self.taMins, self.taMaxs, l)
    local dTempErr = self:CrossValidation(oM, teInput, teTarget, nKFold, l)
    if dTempErr < dBestErr then
      dBestErr = dTempErr
      dBestLambda = l
    end
  end
  
  -- train with bestL
  self.oM = CMLinearL2.new(self.nInputs, self.taMins, self.taMaxs, dBestLambda)
  self.oM:train(teInput, teTarget)
  self.teTheta:copy(self.oM.teTheta)
end
