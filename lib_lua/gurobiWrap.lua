--[[ Description: Utility functions to wrap GUROBI for GNN's usage of linear programming. 
]]

local gurobi = require 'gurobi'
do
	local gurobiWrap = {}
	
	-- metatable, used for final cleanup, *** this requires lua 5.2, this is needed for running on cluster (given the shared environment, to reuse the same environment instead of creating/freeing multiple ones, particularly upon failure)
	gurobiWrap.mt = { 
                env =  gurobi.loadenv(""),
                __gc = function(self) 
		        gurobi.free(self.mt.env, nil)
                       end
		}
	setmetatable(gurobiWrap, gurobiWrap.mt)

	-- Description, solves Aw = y constrained with  nLStart, nLLength to define the weights (w) to be nonnegative
	function gurobiWrap.gelsNonNegative(teA, teY, nLStart, nLLength)
		--local env = gurobi.loadenv("")

		local nD = teA:size(2)
		local nM = teA:size(1)

		local teC = torch.ones(nD + nM * 2)
		teC:narrow(1, 1, nD):fill(0)

		local teG = torch.cat({teA, torch.eye(nM), -torch.eye(nM)}, 2)

		local dVeryNegative=0
		local teLB = torch.Tensor(teC:size()):fill(dVeryNegative)

		if nLStart and nLLength then -- if constraints provided
			teLB:narrow(1, nLStart, nLLength):fill(0)
		end

		teLB:narrow(1, nD + 1, nM * 2):fill(0)

		local model = gurobi.newmodel(gurobiWrap.mt.env, "", teC, teLB)
		gurobi.addconstrs(model, teG, 'EQ', teY)

		-- solve
		local status, teW = gurobi.solve(model)

		local teWCopy = teW:narrow(1, 1, nD):clone()
		gurobi.free(nil, model)

		return status, teWCopy
	end
  
  -- Description, solves Aw = y constrained with  all weights as nonnegative. teAs, teYs constrain the model as various point to have teYs be less than 1. 
	function gurobiWrap.gelsNonNegativeWithMax(teA, teY, teAs, teYs)
		--local env = gurobi.loadenv("")

		local nD = teA:size(2)
		local nM = teA:size(1)
    local nMs = teAs:size(1)

    local dAlpha = 0.01 -- Regularization Parameter
		local teC = torch.Tensor(nD + nM * 2 + nMs):fill(dAlpha)
		teC:narrow(1, nD + 1, nM * 2):fill(1)

		local teG = torch.cat({teA, torch.eye(nM), -torch.eye(nM), torch.zeros(nM, nMs)}, 2) -- for data
    local teGs = torch.cat({teAs, torch.zeros(nMs, 2*nM), torch.eye(nMs)}, 2) -- for max constraints
    
    -- combine:
    teG = torch.cat({teG, teGs}, 1)
    teY = torch.cat({teY, teYs}, 1)

		local teLB = torch.Tensor(teC:size()):fill(0) -- all parameters should be Greater than zero

		local model = gurobi.newmodel(gurobiWrap.mt.env, "", teC, teLB)

		gurobi.addconstrs(model, teG, 'EQ', teY)
		-- solve
		local status, teW = gurobi.solve(model)

		local teWCopy = teW:narrow(1, 1, nD):clone()
		gurobi.free(nil, model)

		return status, teWCopy
	end
  

	return gurobiWrap

end
