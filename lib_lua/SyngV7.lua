--[[ Description: Utility functions to train the activation function of GNN. 
]]

local autograd = require 'autograd'
local gurobiW = require('./gurobiWrap.lua')

local SyngV7 = {}

do
	SyngV7.fuSyngV7 = function(teX, teTheta, bias)
		local nD = teX:size(2)
		local nM = teX:size(1)

		local teW = torch.narrow(teTheta, 1,  1, 2*nD+1)

		local teT = torch.narrow(teW, 1, 1, nD+1)
		local teB = torch.narrow(teW, 1, nD+2, nD)
		local teP = torch.narrow(teTheta, 1, teW:size(1)+1, nD)

		local teH = torch.exp(torch.cmul(teX,
																				torch.expand(torch.view(teP, 1, nD), 
																										 nM, nD)))

		local teTop = torch.bmm(torch.view(torch.cat(torch.ones(nM, 1), teH), 1, teH:size(1), teH:size(2)+1),
														torch.view(teT,1, teH:size(2)+1, 1))

		local teBut = torch.add(torch.bmm(torch.view(teH, 1, teH:size(1), teH:size(2)),
																			torch.view(teB, 1, teH:size(2), 1)), 1)

		local teR = torch.view(torch.cdiv(teTop, teBut), nM, 1)

		return teR
	end

	function SyngV7.getMSE(teX, teTheta, teY)
		local teYPred = SyngV7.fuSyngV7(teX, teTheta)
		return (teY - teYPred):pow(2):mean()
	end

	-- ***  getInitWeights related functions (includes optimization): ***
	
	function SyngV7.getOptimGels(teX, teY, teP)
		-- goal: argMin |Ax-B|
		-- b) construct A
		local nD = teX:size(2)
		local nM = teX:size(1)
		local teH = torch.exp(torch.cmul(teX,
																		torch.expand(torch.view(teP, 1, nD), 
																								 nM, nD)))

		local teYH = torch.cmul(torch.expand(torch.view(teY, nM, 1), nM, nD),
														teH)

		local teA = torch.cat({torch.ones(nM, 1),
													 teH,
													 torch.mul(teYH, -1)})

		-- c) construct B
		local teB = torch.Tensor(nM) -- Note: 2d if real gels used
		teB:copy(teY)
		
--		local teW = torch.gels(teB, teA):squeeze() --ToDo: remove this, it's  old Line when using actual gels
		local nLStart = teA:size(2) - nD
		local status, teW = gurobiW.gelsNonNegative(teA, teB, nLStart, nD)
		if status ~= 2 then
			io.write(string.format(" !!! status:%d !!!! ", status))
		end

		return teW
	end

	function getRandomInput(nRandomSamples, teMins, teMaxs)
		local nD = teMins:size(1)
		local teX = torch.rand(nRandomSamples, nD)
		local teRangeExpanded = torch.Tensor(1, nD):copy(teMaxs-teMins):expand(nRandomSamples, nD)
		local teMinsExpanded = torch.Tensor(1, nD):copy(teMins):expand(nRandomSamples, nD)
		teX = torch.cmul(teX, teRangeExpanded)
		teX = torch.add(teX, teMinsExpanded)
		teX = torch.cat({torch.Tensor(1, nD):copy(teMins), torch.Tensor(1, nD):copy(teMaxs), teX}, 1)

		return teX
	end

	function SyngV7.getOptimGelsWithMax(teX, teY, teP, taMins, taMaxs)
		-- goal: argMin |Ax-B|
		-- b) construct A
		local nD = teX:size(2)
		local nM = teX:size(1)
		local teH = torch.exp(torch.cmul(teX,
																		torch.expand(torch.view(teP, 1, nD), 
																								 nM, nD)))

		local teYH = torch.cmul(torch.expand(torch.view(teY, nM, 1), nM, nD),
														teH)

		local teA = torch.cat({torch.ones(nM, 1),
													 teH,
													 torch.mul(teYH, -1)})

		-- c) construct B
		local teB = torch.Tensor(nM) -- Note: 2d if real gels used
		teB:copy(teY)
		
	    -- ** assuming max of "1"
	    -- d) construct teXs
	    local teXs = getRandomInput(100, taMins.inputs, taMaxs.inputs) -- nRandomSamples:200
	    
	    -- e) construct As
	    nM = teXs:size(1)
	    teH = torch.exp(torch.cmul(teXs,
																		torch.expand(torch.view(teP, 1, nD), 
																								 nM, nD)))
		teYH = torch.mul(teH, taMaxs.output)

		local teAs = torch.cat({torch.ones(nM, 1),
													  teH,
													  torch.mul(teYH, -1)})
    
	    -- f) construct Bs
	    local teBs = torch.Tensor(nM):fill(taMaxs.output)
    
		local status, teW = gurobiW.gelsNonNegativeWithMax(teA, teB, teAs, teBs)
		if status ~= 2 then
			io.write(string.format(" !!! status:%d !!!! ", status))
		end

		return teW
	end
	
	function SyngV7.fuLoss(teP, teW, teX, teY)
		local nD = teX:size(2)
		local nM = teX:size(1)
		-- Loss function: |Aw-B| ToDo:consider the original cost function

		-- a) construct A
		local teH = torch.exp(torch.cmul(teX,
																				torch.expand(torch.view(teP, 1, nD), 
																										 nM, nD)))
		local teYH = torch.cmul(torch.expand(torch.view(teY, nM, 1), nM, nD),
														teH)

		local teA = torch.cat({torch.ones(nM, 1),
													 teH,
													 torch.mul(teYH, -1)})

		-- c) construct B
		local teB = torch.Tensor(nM, 1)
		teB:select(2, 1):copy(teY)
		
		local teResBase = torch.add(torch.bmm(torch.view(teA, 1, teA:size(1), teA:size(2)), 
																					torch.view(teW, 1, teW:size(1), 1)),
															 torch.mul(teB, -1))

		local dRes = torch.bmm(torch.transpose(teResBase, 2, 3), teResBase)[1][1][1]

		--adding regularization here:
		--todo: try this with fuLoss2 instead:
		local dLambda= 0.00
		dRes = dRes + dLambda * torch.sum(torch.pow(teP, 2)) 

		return dRes

	end

	function isValid(teX)
		if teX:max() < 20 then
		  return true
		else
		  return false
		end
	end
  
	function SyngV7.fuForOptim(teX, teY, teInitParam, taMins, taMaxs) -- ToDo: test(although minmal code)
	    if not isValid(teInitParam) then -- ** THIS IS A HACK BUT Needed to guard rare big values, not sure why CG comes up with!
	      teInitParam:fill(0)
	      io.write("!! GUARD USED !!")
	    end
    
		-- a) assume const teInitParam (i.e. p1, p2) and get teW
		local teW = SyngV7.getOptimGelsWithMax(teX, teY, teInitParam, taMins, taMaxs)

		-- b) assume const teW and calculate grad for p1, p2
		local fuGradLoss = autograd(SyngV7.fuLoss)
		local teGradParams, loss = fuGradLoss(teInitParam, teW, teX, teY)

		return loss, teGradParams, teW
	end

  	function  SyngV7.getInitWeightsFast(teX, teY)
    	local nD = teX:size(2)
    	local teTheta = torch.Tensor(nD*3+1):fill(0)
    	teTheta[1] = teY:mean()
    	return teTheta
  	end

	function SyngV7.getInitWeights(teX, teY, taMins, taMaxs, oParamCache) -- ToDo: test
		local nD = teX:size(2)


		local teWeightOptim = nil
		local fuEval = function(teParam)
			local loss, teGradParams, teCurrWeights = SyngV7.fuForOptim(teX, teY, teParam, taMins, taMaxs)
    		teWeightOptim = teCurrWeights:clone()
			return loss, teGradParams
		end

		-- OuterLoop for multiple initializations
		local nMaxRounds = 10 --10
		local teBestTheta = torch.Tensor(nD*3+1)
		local dBestErr = math.huge
    	local dMinGoodErr = (teY - teY:mean()):pow(2):mean()
    	local dGoodEnoughErr = dMinGoodErr/20

		local nCount = 0
      	local teTmp
    	for r=1, nMaxRounds do
      		nCount = nCount + 1

			local teInitParam =  oParamCache:get(nD)
         	teTmp = teInitParam:clone()
			local teParamOptim, lossOptim 
			for i=1, 1 do
				teParamOptim, lossOptim = optim.cg(fuEval, teInitParam)
				--teInitParam = teParamOptim
			end
			local teCurrTheta = torch.cat(teWeightOptim, teParamOptim, 1)

			local dCurrErr = SyngV7.getMSE(teX, teCurrTheta, teY)
			if dCurrErr < dBestErr then
				io.write(string.format("(%f).", dCurrErr))
				dBestErr = dCurrErr
				teBestTheta:copy(teCurrTheta)
			end

			-- Early Stop Criteria
			if dBestErr < dGoodEnoughErr then
				io.write(" *^* ")
				break
			end
		end
  
  		io.write("=")
  		io.write(dBestErr .. "=")
--   for i=1, teTmp:size(1) do
--     io.write(string.format("%.4f_", teTmp[i]))
--   end

  		io.write("=(#" .. nCount .. ")=")
  
  		local teBestP = teBestTheta:narrow(1, nD*2 + 2, nD)
  		if dBestErr > dMinGoodErr then
		    teBestTheta:fill(0)
		    teBestTheta[1] = teY:mean()
		    dBestErr = SyngV7.getMSE(teX, teBestTheta, teY) 
		    io.write("** fail back to mean, MSE: **" .. dBestErr)
		    dBestErr = nil
  		end


		return teBestTheta, teBestP, dBestErr
	end

	return SyngV7
end
