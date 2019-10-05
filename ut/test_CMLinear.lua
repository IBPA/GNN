--[[ Description:  Test functionality of all functions in the CMLinear Model
]]

require 'nn'
require('../lib_lua/CMLinear.lua')
local testSuite = torch.TestSuite()

local tester = torch.Tester()

-- Creates instance of CMLinear and call initial
function testSuite.initial()
  -- Prepare Inputs and Outputs
  local nD = 2
  local taMins = { output = 0, inputs = torch.zeros(nD) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }

  -- Create Module
  local mNet = CMLinear.new(nD, taMins, taMaxs)

  -- Validate
  local nExpectedMulTerms = (nD * (nD-1))/2
  local teExpectedTheta = torch.zeros(1, nD + nExpectedMulTerms + 1)
  local teExpectedGradTheta = torch.zeros(teExpectedTheta:size())

  tester:eq(mNet.nInputs, nD, 0.001, "nInputs should match expected value.")
  tester:eq(mNet.taMins, taMins, 0.001, "taMins should match expected value.")
  tester:eq(mNet.taMaxs, taMaxs, 0.001, "taMaxs should match expected value.")
  tester:eq(mNet.nMulTerms, nExpectedMulTerms, 0.001, "nMulTerms should match expected value.")
  tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
  tester:eq(mNet.teGradTheta, teExpectedGradTheta, 0.001, "teGradTheta should match expected value.")
end

-- Creates instance of CMLinear and call predict
function testSuite.predict()
   -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1, 2}, 
                                 {2, 3},
                                 {1, 4}})

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinear.new(nD, taMins, taMaxs)

   -- Predict
   local teOutput = mNet:predict(teInput)

   -- Validate
   local teExpectedOutput = torch.Tensor({{0.0000}, {0.0000}, {0.0000}})
   tester:eq(teOutput, teExpectedOutput, 0.001, "teOutput should match expected value.")
end

-- Creates instance of CMLinear and call train
-- this train is with 2D input, thus 2+1+1 = 4d for the theta. Careful: input matrix must be non-singular (invertible).
function testSuite.train_2D()
   -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1, 2}, 
                                 {2, 3},
                                 {1, 4},
                                 {3 ,4}})
   local teTarget = torch.Tensor({{1}, 
                                  {2},
                                  {3},
                                  {4}})

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinear.new(nD, taMins, taMaxs)

   -- Train
   mNet:train(teInput, teTarget)


   -- Validate
   local teExpectedTheta = torch.Tensor({{0.5000, -1.5000,  0.5000,  0.5000}})
   tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinear and call train
-- this train is with 1D input, thus 1+0+1 = 2d for the theta. Careful: input matrix must be non-singular (invertible).
function testSuite.train_1D()
   -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1},
                                  {2}})
   local teTarget = torch.Tensor({{1}, 
                                  {3}})

   -- Create Module
   local nD = 1
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinear.new(nD, taMins, taMaxs)

   -- Train
   mNet:train(teInput, teTarget)

   -- Validate
   local teExpectedTheta = torch.Tensor({{-1.0000,  2.0000}})
   tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinear and call getParamPointer
function testSuite.getParamPointer()
   -- Prepare Inputs and Outputs

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinear.new(nD, taMins, taMaxs)

   -- Validate
   local teExpectedTheta = torch.Tensor({{0.0000, 0.0000, 0.0000, 0.0000}})
   local teTheta = mNet:getParamPointer()
   tester:eq(teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

-- Creates instance of CMLinear and call getGradParamPointer
function testSuite.getGradParamPointer()
   -- Prepare Inputs and Outputs

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinear.new(nD, taMins, taMaxs)

   -- Validate
   local teExpectedGradTheta = torch.Tensor({{0.0000, 0.0000, 0.0000, 0.0000}})
   local teGradTheta = mNet:getGradParamPointer()
   tester:eq(teGradTheta, teExpectedGradTheta, 0.001, "teGradTheta should match expected value.")
end

-- example test 
function testSuite.test_Example_Validate()
  local a = {2, torch.Tensor{1, 2, 2}}
  local b = {2, torch.Tensor{1, 2, 2.001}}
  tester:eq(a, b, 0.01, "a and b should be approximately equal")
end

tester:add(testSuite)
tester:run()
