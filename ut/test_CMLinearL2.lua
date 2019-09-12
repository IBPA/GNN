--[[ Description:  Test functionality of all functions in the CMLinearL2 Model
]]
require 'nn'
require('../lib_lua/CMLinearL2.lua')
local testSuite = torch.TestSuite()

local tester = torch.Tester()

-- Creates instance of CMLinear and call initial
function testSuite.initial()
  -- Prepare Inputs and Outputs
  local nDimension = 2
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local dLambda = 0

  -- Create Module

  local mNet = CMLinearL2.new(nDimension, taMins, taMaxs, dLambda)

  -- Validate
  local nExpectedMulTerms = (nDimension * (nDimension-1))/2
  local teExpectedTheta = torch.zeros(1, nDimension + nExpectedMulTerms + 1)
  local teExpectedGradTheta = torch.zeros(teExpectedTheta:size())

  tester:eq(mNet.nInputs, nDimension, 0.001, "nInputs should match expected value.")
  tester:eq(mNet.taMins, taMins, 0.001, "taMins should match expected value.")
  tester:eq(mNet.taMaxs, taMaxs, 0.001, "taMaxs should match expected value.")
  tester:eq(mNet.dlambda, dLambda, 0.001, "dLambda should match expected value.")
  tester:eq(mNet.nMulTerms, nExpectedMulTerms, 0.001, "nMulTerms should match expected value.")
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
  tester:eq(mNet.teGradTheta, teExpectedGradTheta, 0.001, "teGradTheta should match expected value.")
end

-- Creates instance of CMLinear and call train
-- Training with 2D input, thus will output 2+1(extend)+1(bias) = 4D theta.
function testSuite.train_2D()
  -- Prepare Inputs and Outputs
  local teInput = torch.Tensor({{1, 2}, 
                                {2, 3},
                                {3, 5},
                                {5, 7}})
  local teTarget = torch.Tensor({{1}, 
                                {2},
                                {3},
                                {4}})

  -- Create Module
  local nDimension = 2
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local dLambda = 0
  local mNet = CMLinearL2.new(nDimension, taMins, taMaxs, dLambda)

  -- Train
  mNet:train(teInput, teTarget)

  -- Validate
  local teExpectedTheta = torch.Tensor({{-0.5833, 0.9167, 0.4167, -0.0833}})
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinear and call train
-- Training with 1D input, thus will output 1+0(extend)+1(bias) = 2D theta.
function testSuite.train_1D()
  -- Prepare Inputs and Outputs
  local teInput = torch.Tensor({{1},
                                {2}})
  local teTarget = torch.Tensor({{1}, 
                                {3}})

  -- Create Module
  local nDimension = 1
  local taMins = { output = 0, inputs = torch.zeros(nDimension) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nDimension):fill(10) }
  local dLambda = 0
  local mNet = CMLinearL2.new(nDimension, taMins, taMaxs, dLambda)

  -- Train
  mNet:train(teInput, teTarget)

  -- Validate
  local teExpectedTheta = torch.Tensor({{-1.0000,  2.0000}})
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

tester:add(testSuite)
tester:run()