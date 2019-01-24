--[[ Description: Utility functions to train the activation function of Linear GNN. 
]]

local autograd = require 'autograd'
local gurobiW = require('./gurobiWrap.lua')

local SyngLinear = {}

do
	SyngLinear.fuSyngV7 = function(teX, teW)
		local nM = teX:size(1)
		local nD = teX:size(2) + 1
		local teXp = torch.cat(torch.ones(nM, 1), teX)
		local batch1 = torch.view(teXp, 1, nM, nD)
		local batch2 = torch.view(teW, 1, nD, 1)
		local teMul =  torch.view(torch.bmm(batch1, batch2), nM, 1)
		return teMul
	end

	function SyngLinear.getMSE(teX, teW, teY)
		local teYPred = SyngV7.fuSyngV7(teX, teW)
		return (teY - teYPred):pow(2):mean()
	end

	-- ***  getInitWeights related functions (includes optimization): ***

  function  SyngLinear.getInitWeights(teX, teY)
	  local nM = teX:size(1)
	  local teXp = torch.cat(torch.ones(nM, 1), teX)
    local teW = torch.gels(teXp, teY):t():contiguous()

    return teW
  end

	return SyngLinear
end
