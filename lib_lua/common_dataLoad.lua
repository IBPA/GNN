--[[ Description: Utility functions used for loading data.
]]

local csv = csv or require("csv")
local myUtil = myUtil or require('./common_util.lua')

local dataLoad = {}

do
  function dataLoad.pri_getSortedKeysTable(taInput)
    local taRes = {}
    for k, v in pairs(taInput) do
      table.insert(taRes, k)
    end

    table.sort(taRes)

    return taRes
  end

	-- Input: csv file name
	-- Output: table containing key, value pairs as in the .csv file
	function dataLoad.loadTaSetting(strFilename)
		local taSetting = {}
    local taLoadParams = {header=false, separator=","}
    local f = csv.open(strFilename, taLoadParams)
    for fields in f:lines() do
			local strKey = fields[1]
			local strValue = fields[2]
			taSetting[strKey] = strValue
		end

		return taSetting
	end

  function dataLoad.getHeader(strFilename)
print(strFilename)
    local taLoadParams = {header=false, separator="\t"}
    local f = csv.open(strFilename, taLoadParams)
    local fields = f:lines()()
    local nGenes =  #fields

    local taGenes = {}
    for i=1, nGenes do
      table.insert(taGenes, fields[i])
    end

    return taGenes
  end

  function dataLoad.pri_getTableFromArray(taInput)
    local taRes = {}
    for k, v in pairs(taInput) do
      taRes[v]=true
    end

    return taRes
  end
  
  function  dataLoad.getIdsFromRow(strFilename, nRowId)
    local taLoadParams = {header=false, separator=","}
    local f = csv.open(strFilename, taLoadParams)
    local i = 1
    for fields in f:lines() do
      if i == nRowId then
        return torch.LongTensor(fields):add(1) -- add one to since it's based on zero offset
      end
      i = i + 1
    end
  end

  function dataLoad.pri_loadTableOfTensorsFromTsv_Header(taParam)
    local strFilename = taParam.strFilename
    local nCols = taParam.nCols
    local taColsTable = dataLoad.pri_getTableFromArray(taParam.taCols)

    local taLoadParams = {header=false, columns=taColsTable, separator="\t"}
    local f = csv.open(strFilename, taLoadParams)

    local taData= {}
    for fields in f:lines() do
      if type(fields) == "table" and next(fields) ~= nil then
        local teRow = torch.Tensor(nCols)
        local nColId = 0

        for k, strGeneName in pairs(taParam.taCols) do
          if fields[strGeneName] ~= nil and string.len(fields[strGeneName]) > 0  then
            nColId = nColId + 1
            teRow[nColId] = fields[strGeneName]
          end
        end

        if nColId > 0 then
          table.insert(taData, teRow:clone())
        end

      end
    end

    return taData
  end

  function dataLoad.pri_loadTableOfTensorsFromTsv_noHeader(taParam)
    local strFilename = taParam.strFilename
    local nCols = taParam.nCols

    local taLoadParams = {header=false, separator="\t"}
    local f = csv.open(strFilename, taLoadParams)

    local taData= {}
    for fields in f:lines() do
      local teRow = torch.Tensor(nCols)
      for i=1, nCols do
        teRow[i] = tonumber(fields[i]) 
      end

      table.insert(taData, teRow)
    end

    return taData
  end

  function dataLoad.loadTensorFromTsv(taParam)
    local isHeader = taParam.isHeader or false

    local taData = nil

    if isHeader then
      taData = dataLoad.pri_loadTableOfTensorsFromTsv_Header(taParam)
    else
      taData = dataLoad.pri_loadTableOfTensorsFromTsv_noHeader(taParam)
    end

    return myUtil.getTensorFromTableOfTensors(taData)
  end

  function dataLoad.getMaskedSelect(teInput, teMaskDim1)
    local nRows = teInput:size(1)

    -- expand tensor for maskedCopy
    local teMaskSize = teInput:size():fill(1)
    teMaskSize[1] = nRows
    local teMask = torch.ByteTensor(teMaskSize)

    local teInputMasked = nil
    if teInput:dim() == 2 then
      teMask:select(2, 1):copy(teMaskDim1)
      teMask = teMask:expandAs(teInput)
      teInputMasked = teInput:maskedSelect(teMask)
      teInputMasked:resize(teMaskDim1:sum(), teInput:size(2))

    elseif teInput:dim(2) == 3 then
      teMask:select(3, 1):select(2, 1):copy(teMaskDim1)
      teMask = teMask:expandAs(teInput)
      teInputMasked = teInput:maskedSelect(teMask)
      teInputMasked:resize(teMaskDim1:sum(), teInput:size(2), teInput:size(3))

    else
      error(string.format("nDim = %d not supported!", teInput:dim()))
    end

    return teInputMasked
  end


  function dataLoad.pri_getMasked(teData, teMask)
    local nSize = teData:size(1)
    local nDataWidth = teData:size(2)
    local teDataMask = torch.repeatTensor(teMask, nDataWidth, 1):t()
    local teResult = teData:maskedSelect(torch.ByteTensor(nSize, nDataWidth):copy(teDataMask))
    teResult:resize(teMask:sum(), nDataWidth)

    return teResult
  end

  function dataLoad.loadTrainTest(taParam, nFolds) -- ToDo: not individually tested!
    nFolds = nFolds or 2
    local teInput = dataLoad.loadTensorFromTsv(taParam.taInput)
    local teTarget = dataLoad.loadTensorFromTsv(taParam.taTarget)

    local nSize = teInput:size(1)
    local teIdx = torch.linspace(1, nSize, nSize)

    -- train:
    local trainMask = torch.mod(teIdx, nFolds):eq(torch.zeros(nSize))
    local teTrain_input = dataLoad.pri_getMasked(teInput, trainMask)
    local teTrain_target = dataLoad.pri_getMasked(teTarget, trainMask)
    local taTrain = {teTrain_input, teTrain_target}


    -- test
    local testMask = torch.mod(teIdx, nFolds):ne(torch.zeros(nSize))
    local teTest_input = dataLoad.pri_getMasked(teInput, testMask)
    local teTest_target = dataLoad.pri_getMasked(teTarget, testMask)
    local taTest = {teTest_input, teTest_target}

    return taTrain, taTest

  end

  return dataLoad
end
