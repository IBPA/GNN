require 'nn'
require('../lib_lua/CMLinearL2AutoLambda.lua')
local testSuite = torch.TestSuite()

local tester = torch.Tester()

-- Creates instance of CMLinearL2AutoLambda and call initial
function testSuite.initial()
  -- Prepare Inputs and Outputs
  local nD = 2
  local taMins = { output = 0, inputs = torch.zeros(nD) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }

  -- Create Module

  local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)


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


-- Creates instance of CMLinearL2AutoLambda and call predict
function testSuite.predict()
   -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1, 2}, 
                                 {2, 3},
                                 {1, 4}})

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- Predict
   local teOutput = mNet:predict(teInput)

   -- Validate
   local teExpectedOutput = torch.Tensor({{0.0000}, {0.0000}, {0.0000}})
   tester:eq(teOutput, teExpectedOutput, 0.001, "teOutput should match expected value.")
end

-- Creates instance of CMLinearL2AutoLambda and call train
-- Creates instance of CMLinearL2AutoLambda and call train
-- this train is 2D input, thus 2+1+1 = 4d for the theta. Carefule: input matrix must be non-singular (invertible).
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
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- Train
   mNet:train(teInput, teTarget)

   -- Validate
   local teExpectedTheta = torch.Tensor({{-0.1418, 0.4965, 0.4413, -0.0412}})
   tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end

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
   local nD = 1
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- Train
   mNet:train(teInput, teTarget)

   -- Validate
   local teExpectedTheta = torch.Tensor({{0,  1.0000}})
   tester:eq(mNet.teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end


-- Creates instance of CMLinearL2AutoLambda and call getParamPointer
function testSuite.getParamPointer()
   -- Prepare Inputs and Outputs

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- Validate
   local teExpectedTheta = torch.Tensor({{0.0000, 0.0000, 0.0000, 0.0000}})
   local teTheta = mNet:getParamPointer()
   tester:eq(teTheta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end


-- Creates instance of CMLinearL2AutoLambda and call getGradParamPointer
function testSuite.getGradParamPointer()
   -- Prepare Inputs and Outputs

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- Validate
   local teExpectedGradTheta = torch.Tensor({{0.0000, 0.0000, 0.0000, 0.0000}})
   local teGradTheta = mNet:getGradParamPointer()
   tester:eq(teGradTheta, teExpectedGradTheta, 0.001, "teGradTheta should match expected value.")
end

 function testSuite.test_Example_Validate()
    local a = {2, torch.Tensor{1, 2, 2}}
    local b = {2, torch.Tensor{1, 2, 2.001}}
    tester:eq(a, b, 0.01, "a and b should be approximately equal")
 end

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
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

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

function testSuite.getBeta()
     -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1, 1, 2, 2}, 
                                 {1, 2, 3, 6},
                                 {1, 1, 4, 4},
                                 {1, 3 ,4, 12}})
   local teTarget = torch.Tensor({{1}, 
                                  {2},
                                  {3},
                                  {4}})
   local l = 0

   -- Create Module
   local nD = 2
   local taMins = { output = 0, inputs = torch.zeros(nD) }
   local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
   local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

   -- run function
   local beta = mNet:getBeta(l, teInput, teTarget)
   -- print(beta)


   -- Validate
   local teExpectedTheta = torch.Tensor({{0.5000}, {-1.5000},  {0.5000},  {0.5000}})
   -- print(teExpectedTheta)
   tester:eq(beta, teExpectedTheta, 0.001, "teTheta should match expected value.")
end


-- test get Cross Validation Error function
function testSuite.getCVErr()
  -- Prepare Inputs and Outputs and other parameter
  local teInput = torch.Tensor({{1, 1, 2, 2}, 
                               {1, 2, 3, 6},
                               {1, 3, 5, 15},
                               {1, 5 ,7, 35},
                               {1, 7, 11, 77}})
  local teTarget = torch.Tensor({{1}, 
                                {2},
                                {3},
                                {4},
                                {5}})
  local l = 0
  local kfold = 5

  -- Create Module
  local nD = 2
  local taMins = { output = 0, inputs = torch.zeros(nD) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)

  -- run function
  local resultError = mNet:getCVErr(l, teInput, teTarget, kfold)

  -- validate
  local expectedError = 0.240
  tester:eq(resultError, expectedError, 0.001, "the result error should match expected value.")
end

-- test getMSE
function testSuite.getMSE()
  -- Prepare Inputs and Outputs and other parameter
  local teInput = torch.Tensor({{1, 1},
                                {1, 2}})
  local teTheta = torch.Tensor({{-1.0000}, {2.0000}})
  local teTarget = torch.Tensor({{1}, 
                                  {3}})
  -- Create Module
  local nD = 2
  local taMins = { output = 0, inputs = torch.zeros(nD) }
  local taMaxs = { output = 1, inputs = torch.Tensor(nD):fill(10) }
  local mNet = CMLinearL2AutoLambda.new(nD, taMins, taMaxs)


  -- run fucntion
  local MSE = mNet:getMSE(teInput, teTheta, teTarget)


  -- validate
  local expectedMSE = 0
  tester:eq(MSE, expectedMSE, 0.001, "the MSE should match expected value.")
end
-- Example: Run single test only while developing new tests:
-- testSuite.getBeta()

tester:add(testSuite)
tester:run()