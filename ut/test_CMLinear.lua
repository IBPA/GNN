require 'nn'
require('../lib_lua/CMLinear.lua')
local testSuite = torch.TestSuite()

local tester = torch.Tester()

-- Creates instance of CMLinear and call train
function testSuite.train_only()
   -- Prepare Inputs and Outputs
   local teInput = torch.Tensor({{1, 2}, 
                                 {2, 3},
                                 {1, 4},
                                 {4, 3}})
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

   -- ToDo: Validate parameters next
end

 function testSuite.test_Example_Validate()
    local a = {2, torch.Tensor{1, 2, 2}}
    local b = {2, torch.Tensor{1, 2, 2.001}}
    tester:eq(a, b, 0.01, "a and b should be approximately equal")
 end

-- Example: Run single test only while developing new tests:
-- testSuite.train_only()

tester:add(testSuite)
tester:run()
