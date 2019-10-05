--[[ Description:  Test functionality of all functions in the CMLinearL2AutoLambda Model
]]
require 'nn'
require('../lib_lua/CMLinearL2AutoLambda.lua')
local testSuite = torch.TestSuite()

local tester = torch.Tester()

-- Creates instance of CMLinearL2AutoLambda and call train
-- Training with 2D input, thus will output 2+1(extend)+1(bias) = 4D theta.
function testSuite.train_2D()
  -- Prepare Inputs and Outputs
  local teInput = torch.Tensor({{1, 2}, 
                               {2, 3},
                               {3, 5},
                               {5 ,7},
                               {7, 11}})
  local teTarget = torch.Tensor({{1}, 
                                {2},
                                {3},
                                {4},
                                {5}})

  -- Create Module
  local nDimension = 2
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nDimension, taMins, taMaxs)

  -- Train
  mNet:train(teInput, teTarget)

  -- Validate
  local teExpectedTheta = torch.Tensor({{-0.3992, 0.5282, 0.5202, -0.0524}})
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinearL2AutoLambda and call train
-- Training with 1D input, thus will output 1+0(extend)+1(bias) = 2D theta.
function testSuite.train_1D()
  -- Prepare Inputs and Outputs
  local teInput = torch.Tensor({{1},
                                {2},
                                {3},
                                {4},
                                {5}})
  local teTarget = torch.Tensor({{1}, 
                                {2},
                                {3},
                                {4},
                                {5}})

  -- Create Module
  local nDimension = 1
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nDimension, taMins, taMaxs)

  -- Train
  mNet:train(teInput, teTarget)

  -- Validate
  local teExpectedTheta = torch.Tensor({{0,  1.0000}})
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinearL2AutoLambda and call maskForKFold to mask the input for K-fold Cross Validation
function testSuite.maskForKFold()
  -- prepare input and output
  local teInput = torch.Tensor({{1},
                                 {2},
                                 {3},
                                 {4}})
  local teTarget = torch.Tensor({{1}, 
                                  {2},
                                  {3},
                                  {4}})
  local foldID = 0
  local nFolds = 2

  -- create Module
  local nDimension = 2
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nDimension, taMins, taMaxs)

  -- validate 
  local teExpectedTrain_input = torch.Tensor({{1}, {3}})
  local teExpectedTrain_target = torch.Tensor({{1}, {3}})
  local teExpectedTest_input = torch.Tensor({{2}, {4}})
  local teExpectedTest_target = torch.Tensor({{2}, {4}})
  teTrain_input, teTrain_target, teTest_input, teTest_target = mNet:maskForKFold(teInput, teTarget, foldID, nFolds)

  tester:eq(teTrain_input, teExpectedTrain_input, 0.001, "teTrain_input should match expected value.")
  tester:eq(teTrain_target, teExpectedTrain_target, 0.001, "teTrain_target should match expected value.")
  tester:eq(teTest_input, teExpectedTest_input, 0.001, "teTest_input should match expected value.")
  tester:eq(teTest_target, teExpectedTest_target, 0.001, "teTest_target should match expected value.")
end

-- Creates instance of CMLinearL2AutoLambda and test get Cross Validation Error function
function testSuite.CrossValidation()
  -- Prepare Inputs and Outputs and other parameter
  local teInput = torch.Tensor({{1, 2}, 
                               {2, 3},
                               {3, 5},
                               {5 ,7},
                               {7, 11}})
  local teTarget = torch.Tensor({{1}, 
                                {2},
                                {3},
                                {4},
                                {5}})
  local dLambda = 0
  local nFolds = 5

  -- Create Module
  local nDimension = 2
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nDimension, taMins, taMaxs)
  local oModule = CMLinearL2.new(nDimension, taMins, taMaxs, dLambda)

  -- run function
  local dMeanError = mNet:CrossValidation(oModule, teInput, teTarget, nFolds)

  -- validate
  local dExpectedMeanError = 6
  tester:eq(dResultError, dExpectedError, 0.001, "the result error should match expected value.")
end

tester:add(testSuite)
tester:run()