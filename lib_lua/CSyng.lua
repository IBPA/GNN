local autograd = require 'autograd'
CSyng = torch.class('CSyng')
local Syng = require('./SyngV7.lua')
--local Syng = require('./SyngLinear.lua')

function CSyng:__init(nInputs, taMins, taMaxs, oParamCache)
  self.nInputs = nInputs
  self.weight = torch.Tensor()
  self.gradWeight = torch.Tensor()
  
  self.taMins = taMins
  self.taMaxs = taMaxs
  self.oParamCache = oParamCache
end

function CSyng:getParamPointer()
  return self.weight
end

function CSyng:getGradParamPointer()
  return self.gradWeight
end

function CSyng:predict(teInput)
  return self.mAutoClamp:forward(teInput)
end

function CSyng:train(teInput, teTarget, isFast)
  local teW, teP, dTrainErr
  if isFast then
    teW = Syng.getInitWeightsFast(teInput, teTarget)
  else
    teW, teP, dTrainErr = Syng.getInitWeights(teInput, teTarget, self.taMins, self.taMaxs, self.oParamCache)
    self.oParamCache:update(teP, dTrainErr)
  end
  
  self.weight:set(teW)
  self.mAuto = autograd.nn.AutoModule('SyngTwoV7')(Syng.fuSyngV7, self.weight)
  self.gradWeight:set(self.mAuto.gradWeight)

  self.mAutoClamp = nn.Sequential()
  self.mAutoClamp:add(self.mAuto)
  self.mAutoClamp:add(nn.Clamp(self.taMins.output, self.taMaxs.output))
  
  return teP, dTrainErr
end

function CSyng:getGradInput(teInput, teGradOutput)
  return self.mAutoClamp:updateGradInput(teInput, teGradOutput)
end

function  CSyng:accGradParameters(teInput, teGradOutput, scale)
  self.mAuto.weight:copy(self.weight)
  self.mAutoClamp:accGradParameters(teInput, teGradOutput, scale)
  self.gradWeight:copy(self.mAuto.gradWeight)
end
