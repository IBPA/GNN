require 'nn'
require 'optim'
local myUtil = myUtil or require('./common_util.lua')

do
  local trainerPool = {}

  function trainerPool.getDefaultTrainParams(nRows, strOptimMethod)

    local taTrainParam = {  batchSize = math.floor(nRows),
                            criterion = nn.MSECriterion(),
                            nMaxIteration = 100,
                            coefL1 = 0.0,
                            coefL2 = 0.0,
                            strOptimMethod = strOptimMethod or "CG",
                            isLog = true,
                            taOptimParams = {}
                          }

    if taTrainParam.strOptimMethod == "SGD" then
      taTrainParam.taOptimParams = { 
        learningRate = 1e-23,
      --learningRateDecay = 0.9995,
        --momentum = 0.5 
        }
      taTrainParam.fuOptim = optim.sgd
  
    elseif taTrainParam.strOptimMethod == "LBFGS" then
      taTrainParam.taOptimParams = { 
        maxIter = 100,
        lineSearch = optim.lswolfe }
      taTrainParam.fuOptim = optim.lbfgs

    elseif taTrainParam.strOptimMethod == "CG" then
      taTrainParam.taOptimParams = {
        maxIter = 20 }
      taTrainParam.fuOptim = optim.cg
    elseif taTrainParam.strOptimMethod == "AdaGrad" then
      taTrainParam.taOptimParams = {
        learningRate = 0.0000000000001 }
      taTrainParam.fuOptim = optim.adagrad
      
    elseif taTrainParam.strOptimMethod == "RMSprop" then
      taTrainParam.taOptimParams = {
        learningRate = 0.0000000000001 }
      taTrainParam.fuOptim = optim.rmsprop

    else
      error("invalid operation")
    end

    return taTrainParam
  end

  function trainerPool.pri_trainGrnn_SingleRound(mNet, taInput, teTarget, taTrainParam)
    parameters, gradParameters = mNet:getParameters()
    local criterion = taTrainParam.criterion
    local overallErr = 0
    local nRows = taInput[1]:size(1)

    for t = 1,nRows, taTrainParam.batchSize do
      -- create batches
      --myUtil.log("batch first item:" .. t, true, taTrainParam.isLog)
      local nCurrBatchSize = math.min(taTrainParam.batchSize, nRows - t + 1)
      
      local taBatchX = {taInput[1]:narrow(1, t, nCurrBatchSize), taInput[2]:narrow(1, t, nCurrBatchSize)}
      local teBatchY = teTarget:narrow(1, t, nCurrBatchSize)

      local fuEval = function(x)
        collectgarbage()

        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()

        -- evaluate function for the complete mini batch
        local teBatchPredY = mNet:forward(taBatchX)
        local f = criterion:forward(teBatchPredY, teBatchY)

        -- estimate df/dW
        local df_do = criterion:backward(teBatchPredY, teBatchY)
        mNet:backward(taBatchX, df_do)

       -- penalties (L1 and L2):
        if taTrainParam.coefL1 ~= 0 or taTrainParam.coefL2 ~= 0 then
          -- locals:
           local norm,sign= torch.norm,torch.sign
 
          -- Loss:
          f = f + taTrainParam.coefL1 * norm(parameters,1)
          f = f + taTrainParam.coefL2 * norm(parameters,2)^2/2

          -- Gradients:
          gradParameters:add( sign(parameters):mul(taTrainParam.coefL1) + parameters:clone():mul(taTrainParam.coefL2) )
        end
        
        overallErr = overallErr + f

        return f, gradParameters
      end --fuEval

      taTrainParam.fuOptim(fuEval, parameters, taTrainParam.taOptimParams)
    end

    return trainerPool.getErr(mNet, taInput, teTarget, taTrainParam)
  end

  function trainerPool.getErr(mNet, taInput, teTarget, taTrainParam)
    local criterion = taTrainParam.criterion or nn.MSECriterion()

    local teOutput = mNet:forward(taInput)
    local fErr = criterion:forward(teOutput, teTarget)

    return fErr
  end

  function trainerPool.trainGrnnMNet(mNet, taInput, teTarget, taTrainerParamsOverride)
    local criterion = nn.MSECriterion()
    local taTrainParam = trainerPool.getDefaultTrainParams(taInput[1]:size(1), taTrainerParamsOverride.strOptimMethod or "CG" )
    myUtil.updateTable(taTrainParam, taTrainerParamsOverride)

    local errPrev = math.huge
    local mNetPrev = nil
    local errCurr = math.huge

		print("optimizing using: " .. taTrainParam.strOptimMethod)
    local dTmpErr = trainerPool.getErr(mNet, taInput, teTarget, taTrainParam)
    print("error before training: " .. dTmpErr)
    for i=1, taTrainParam.nMaxIteration do
      errCurr = trainerPool.pri_trainGrnn_SingleRound(mNet, taInput, teTarget, taTrainParam)

--      if errPrev <= errCurr or myUtil.isNan(errCurr)  then
      if  myUtil.isNan(errCurr)  then
        print("** early stop **")
        return errPrev, mNetPrev
      elseif errCurr ~= nil then
        local message = errCurr < errPrev and "<" or "!>"
				message = message .. errCurr
        myUtil.log(message, false, taTrainParam.isLog)
        errPrev = errCurr
        mNetPrev = mNet:clone()
      else
        error("invalid value for errCurr!")
      end

    end

    return errCurr, mNet
  end

  return trainerPool
end
