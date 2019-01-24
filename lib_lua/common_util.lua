require "cephes"
do
  local myUtil = {}

  -- Input(teData): a tensor in which the first set of columns are input and the rest are output
  -- Input(nInputs): the number of columns in teData associated with input
  -- Input(nOutputs): the number of columns in teData associated with output
  -- Output: returns a table in which, earch row consists of two tensors, the first is for input, the second is for output
  function myUtil.getTableFromTensor(teData, nInputs, nOutputs)
    local train_X = teData:narrow(2, 1, nInputs ) -- take the first two columns as X
    local train_y = teData:narrow(2, nInputs + 1, nOutputs ) -- take the last column as y

    local tableData = {}
    function tableData:size() return teData:size(1) end

    for i=1, teData:size(1) do
      tableData[i] = { train_X[i], train_y[i] }
    end

    return tableData
  end

  -- Input1: Takes a table with 'K' tensors (of the same size 'N' for dimention 1)
  -- Input2: Function used to decide whether to keep a given sample in the Fold
  -- Output: Returns a table with 'K' columns representing the same input data except the ones for which fKeep returns false
  function myUtil.getFoldTableFromTableOfTensors(taData, fKeep)
    local nK = table.getn(taData)
    local nN = taData[1]:size(1)
    local taDataResult = {}

    for n=1, nN do
      local taRow = {}

      if fKeep(n) then
	for k=1, nK do
	  table.insert(taRow, torch.Tensor(taData[k][n]))
	end
	table.insert(taDataResult, taRow)
      end -- if

    end -- for

    local nRows = table.getn(taDataResult)
    function taDataResult:size() return nRows end

    return taDataResult
  end

  -- Input: Takes a table with 'K' tensors (of the same size 'N' for dimention 1)
  -- Output: Returns a table with 'N' rows and 'K' columns representing the same input data
  function myUtil.getTableFromTableOfTensors(taData)
    local nK = table.getn(taData)
    local nN = taData[1]:size(1)
    local taDataResult = {}
    function taDataResult:size() return nN end

    for n=1, nN do
      local taRow = {}
      for k=1, nK do
  --      table.insert(taRow, torch.Tensor(taData[k][n])) -- ToDo: Ensure that the following works in call use cases
	  table.insert(taRow, taData[k]:narrow(1, n, 1))
      end
      taDataResult[n] = taRow
    end

    return taDataResult

  end

  -- Input: Takes a table where each row is a separate tensor.
  -- Output: Returns a single tensor representing the input table
  function myUtil.getTensorFromTableOfTensors(taData)
    -- assuming all tensors are one dimentional and have same size
    local nRows = #taData
    local nCols = taData[1]:size(1)
    local teDataResult = torch.Tensor(nRows, nCols):zero()

    for i=1, nRows do
      teDataResult[i]:copy(taData[i])
    end

    return teDataResult
  end

  --Todo: test this:
  function myUtil.getTensorFromTableOfTensors_col(taData, colId)
    -- assuming all tensors are one dimentional and have same size
    local nRows = table.getn(taData)
    local nCols = taData[1][colId]:size(1)
    local teDataResult = torch.Tensor(nRows, nCols):zero()

    for i=1, nRows do
      teDataResult[i]:copy(taData[i][colId])
    end

    return teDataResult
  end

  function myUtil.pri_addSize(ta)
    function ta:size() return self.n end
  end

  -- Returns a vector representing the provided "dec" number with the "nLength" 
  function myUtil.dec2Tensor(nLength, dec)
    local bTensor = torch.Tensor(nLength):zero()
    
    local rest = dec
    local i=1
    while rest>0 do
      local res = math.fmod(rest, 2)
      bTensor[i] = res
      rest = (rest - res)/2
      i = i+1
    end

    return bTensor
  end

  function myUtil.shallowCopy(original)
      local copy = {}
      for key, value in pairs(original) do
	  copy[key] = value
      end
      return copy
  end

  function myUtil.getPowerVector(base, nLength)
    local seq = torch.range(0, nLength-1)
    local pv = torch.pow(base, seq)
    return pv
  end

  function myUtil.sortDataUsingX(taData)
    local vPV = myUtil.getPowerVector(2, taData[1]:size(2))
    local vUniIds = torch.mv(taData[1], vPV)
    local vSortedUniIds, idx = torch.sort(vUniIds)

    -- expand idx to expId (for X)
    local idx2d = torch.LongTensor(idx:size(1), 1)
    idx2d:select(2, 1):copy(idx)
    local expIdx = torch.expand(idx2d, idx2d:size(1), taData[1]:size(2)) 

    -- reorder x,y  according to idx
    taData[2] = taData[2]:gather(1, idx2d) -- todo: assumes y is 2d, depending on the caller this may varry
    taData[1] = taData[1]:gather(1, expIdx)

    return taData, vSortedUniIds, idx
  end

-- Returns the corresponding value for the given confidence from t-distribution
  function myUtil.getTDistAlpha(nDF, dConfidence)
    return  math.abs(cephes.stdtri(nDF, (1 - dConfidence))) 
  end

  function myUtil.getCsvStringFrom2dTensor(teData, separator)
    local strRes = ""
    separator = separator or ","
    local strFormat = "%.7f"

    for r=1, teData:size(1) do
      for c=1, teData:size(2) - 1 do
        strRes = strRes .. string.format(strFormat, teData[r][c]) .. separator 
      end
        strRes = strRes .. string.format(strFormat, teData[r][teData:size(2)]) .. "\n"
    end

    return strRes
  end

  function myUtil.saveTensorAndHeaderToCsvFile(teData, taHeader, strFilename)
    local fHandle = assert(io.open(strFilename, "w+"))
    local strHeader = table.concat(taHeader, "\t")
    fHandle:write(strHeader .. "\n")
    myUtil.saveCsvStringFrom2dTensorToFile(teData, "\t", fHandle)
    fHandle:close()
  end

  function myUtil.saveCsvStringFrom2dTensorToFile(teData, separator, fHandle)
    separator = separator or ","
    local strFormat = "%.7f"

    for r=1, teData:size(1) do
      for c=1, teData:size(2) - 1 do
        fHandle:write( string.format(strFormat, teData[r][c]) .. separator )
      end
        fHandle:write( string.format(strFormat, teData[r][teData:size(2)]) .. "\n")
    end

  end

  function myUtil.getCsvStringFrom1dTensor(teX, separator)
    local strRes = ""
    separator = separator or ","
    local strFormat = "%.7f"

    for c=1, teX:size(1)-1 do
     strRes = strRes .. string.format(strFormat, teX[c]) .. ","
    end
    strRes = strRes .. string.format(strFormat, teX[teX:size(1)])

    return strRes
  end  

  function myUtil.getCsvStringFromTable(taX, separator)
     local strRes = ""
     separator = separator or ","
    
    for i, taRow in pairs(taX) do
      for k, colValue in pairs(taRow) do
         local strFormat = (type(colValue) == "number") and "%.4f" or "%s"
         strRes = strRes .. string.format(strFormat, colValue) .. ","
      end
      strRes = strRes:sub(1, -2) .. "\n"
    end

    return strRes
  end

  function myUtil.getDeepCopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
      copy = {}
      for orig_key, orig_value in next, orig, nil do
        copy[myUtil.getDeepCopy(orig_key)] = myUtil.getDeepCopy(orig_value)
      end
      setmetatable(copy, myUtil.getDeepCopy(getmetatable(orig)))
    else -- number, string, boolean, etc
      copy = orig
    end
    return copy
  end

  function myUtil.getFilledCurve_identity(min, max, err)
    local x = torch.linspace(min, max)
    local yy = x:clone() 
    yy = torch.cat(yy, x + err, 2)
    yy = torch.cat(yy, x - err, 2)

    return yy
  end

  function myUtil.getNumLines(strFilename)
    local file = assert(io.open(strFilename, "r"))
    local nLines = 0
    for _ in file:lines() do
      nLines = nLines + 1
    end

    file:close()

    return nLines
  end

  function myUtil.isInTaValues(strGene, taTable)
    for k, v in pairs(taTable) do
      if strGene == v then
        return true
      end
    end

    return false
  end

  function myUtil.log(strMessage, isNewLine, isLog)
    if not isLog then
      return
    end

    io.write(strMessage .. (isNewLine and "\n" or ""))
    io.flush()
  end

  function myUtil.isNan(input)
    return input ~= input
  end

  function myUtil.getMNetClone(mNet)
    local mNetClone = mNet:clone()
    return mNetClone

  end

  function myUtil.findIndexInArray(taInput, value)
    for k, v in pairs(taInput) do
      if value == v then
        return k
      end
    end

    return nil
  end

  function myUtil.updateTable(taInput, taUpdate, isCopyTensor)
		if taUpdate == nil then
			return
		end

		if taInput == nil then
			taInput = {}
		end

    for k, v in pairs(taUpdate) do

			if isCopyTensor then
      	taInput[k]:copy(v)
			else
      	taInput[k] = v
			end

    end
  end

  function myUtil.getBoolFromStr(strInput)
    if strInput == nil then
      return false
    end

    return (string.lower(strInput) == "true")
  end

  
  return myUtil

end
