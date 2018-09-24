local dataLoad = dataLoad or require('../../MyCommon/dataLoad.lua')
local myUtil = myUtil or require('../../MyCommon/util.lua')
local csv = csv or require("csv")

data = {}
CData = torch.class('CData')

do
  function data.priGetFilenames(strDir, isNoise, strFoldFilename)
    local strFilenameKO = string.format("%s/processed_KO.tsv", strDir)
    
    local strFilenameNonTF = isNoise and string.format("%s/processed_NonTFs.tsv", strDir) or
                                         string.format("%s/processed_NonTFsNoNoise.tsv", strDir)
                                         
    local strFilenameTF = isNoise and string.format("%s/processed_TFs.tsv", strDir) or
                                         string.format("%s/processed_TFsNoNoise.tsv", strDir)
    
    local strFilenameStratifiedIds = string.format("%s/folds/%s", strDir, strFoldFilename)
                                         
    return strFilenameTF, strFilenameNonTF, strFilenameKO, strFilenameStratifiedIds
  end
  
  function data.priGetData(strFilePath)

    local taGenes = dataLoad.getHeader(strFilePath)
    local taLoadParam = { strFilename = strFilePath, nCols = #taGenes, taCols = taGenes, isHeader = true }
    local teData = dataLoad.loadTensorFromTsv(taLoadParam)

    return { taGenes = taGenes, teData = teData }
  end
  
  function data.priGetDataDepOrder(teData, taColnames, taOrder)
    local teDataReOrdered = torch.Tensor(teData:size())
    for i=1, #taColnames do
      local nColIdOrdered = taOrder[taColnames[i]]
      if nColIdOrdered == nil then
        print("!!! no order info for:".. taColnames[i])
      end
      
      teDataReOrdered:select(2, nColIdOrdered):copy(teData:select(2, i))
    end
    
    return teDataReOrdered
  end
  
  function data.priGetDataOrigOrder(teData, taColnames, taOrder)
    local teDataReOrdered = torch.Tensor(teData:size())
    for i=1, #taColnames do
      local nColIdOrdered = taOrder[taColnames[i]]
      teDataReOrdered:select(2, i):copy(teData:select(2, nColIdOrdered))
    end
    
    return teDataReOrdered
  end
  
  function data.pri_getBalancedIdx(teDataCat, dMinDist)
    print("******")
   local dEpsilon = 1e-10
   local teD = teDataCat:clone()
   local nRows = teD:size(1)
   local nCols = teD:size(2)
   teD = teD - teD:mean(1):expandAs(teD) -- centralize
   teDStd = teD:std(1):expandAs(teD) + dEpsilon -- standardize (add epsilon to avoid possible devision by zero)
   teD = teD:cdiv(teDStd)

   local teM = torch.mm(teD:t(), teD)
   teM:div(nRows-1)

   local teMe, teMV = torch.symeig(teM, 'V')

   local tePC1 = teD*teMV:select(2, nCols)
   local y, idxSort = torch.sort(tePC1, 1, true)
   idxSort = idxSort:squeeze()

    local taIdx = {}
    table.insert(taIdx, idxSort[1])
    local nLastAddedId = idxSort[1]
    for i=2, nRows do
      local dDist = tePC1[nLastAddedId] - tePC1[idxSort[i]]

      if dDist > dMinDist then
        table.insert(taIdx, idxSort[i])
        nLastAddedId = idxSort[i]
      end
    end

    return torch.LongTensor(taIdx)
  end
  
  function data.pri_getSelectedGivenIdx(teData, teIdx)
  local teMask = torch.ByteTensor(teData:size(1)):fill(0)
  teMask:indexFill(1, teIdx, 1)

  return dataLoad.getMaskedSelect(teData, teMask)
end
  
  function data.pri_getBalancedSample(teInputData, teKOData, teTargetData, dMinDist)
    local teDataCat = torch.cat({teInputData, teKOData, teTargetData}, 2)
    local teIdx = data.pri_getBalancedIdx(teDataCat, dMinDist)
    
    return data.pri_getSelectedGivenIdx(teInputData, teIdx),
           data.pri_getSelectedGivenIdx(teKOData, teIdx),
           data.pri_getSelectedGivenIdx(teTargetData, teIdx)
  end
  
  function data.pri_getStratSamples(teInputData, teKOData, teTargetData, teIdx)    
    return data.pri_getSelectedGivenIdx(teInputData, teIdx),
           data.pri_getSelectedGivenIdx(teKOData, teIdx),
           data.pri_getSelectedGivenIdx(teTargetData, teIdx)
  end
  
  function data.priFilterExtraTFs(taTFData, taAll) -- This function is necessary for TFs with nothing to regulate
    local nColsRemain = 0
    local taGenes = {}
    local taTransfer = {}
    for key, value in pairs(taTFData.taGenes) do
      if taAll[value] ~= nil then
        table.insert(taGenes, value)
        nColsRemain = nColsRemain + 1
        taTransfer[key] = nColsRemain
      end
    end
    
    local teData = torch.Tensor(taTFData.teData:size(1), nColsRemain)
  
    for key, value in pairs(taTransfer) do
      teData:select(2, value):copy(taTFData.teData:select(2, key))
    end

    return {taGenes = taGenes, teData = teData}
  end
  
  function CData:priLoad(strDir, oDepGraph, dMinDist, isNoise, strFoldFilename, nStratifiedRowId)
    local strFilenameTF, strFilenameNonTF, strFilenameKO, strFilenameStratifiedIds = data.priGetFilenames(strDir, isNoise, strFoldFilename)
    
    -- 1) Load Input Data
    local taTFData = data.priGetData(strFilenameTF)
    taTFData = data.priFilterExtraTFs(taTFData, oDepGraph.taAll)
    local teInputData = data.priGetDataDepOrder(taTFData.teData, taTFData.taGenes, oDepGraph:getTFOrders())
    
    -- 2) Load Target Data
    local taNonTFData = data.priGetData(strFilenameNonTF)
    self.taNonTFGenes = taNonTFData.taGenes
    local teTargetData = data.priGetDataDepOrder(taNonTFData.teData, taNonTFData.taGenes, oDepGraph:getNonTFOrders())
    
    -- 3) Load KO Data
    local taKOData = data.priGetData(strFilenameKO)
    local teKOData = data.priGetDataDepOrder(taKOData.teData, taKOData.taGenes, oDepGraph:getNonTFOrders())

    -- 4) Extract a subsample (using dMinDist)
    if dMinDist then
      teInputData, teKOData, teTargetData = data.pri_getBalancedSample(teInputData, teKOData, teTargetData, dMinDist)
    else
      local teIdx = dataLoad.getIdsFromRow(strFilenameStratifiedIds, nStratifiedRowId)
      teInputData, teKOData, teTargetData = data.pri_getStratSamples(teInputData, teKOData, teTargetData, teIdx )
    end
    
    return { input = { teInputData, teKOData}, 
              target = teTargetData, 
              strFilenameTF = strFilenameTF, 
              strFilenameNonTF = strFilenameNonTF, 
              strFilenameKO = strFilenameKO }
  end
  
  function loadGERanges(strDir)
    local strFilename = string.format("%s/../../../ge_range.csv", strDir)
    local taLoadParams = {header=true, separator=","}
    local taRanges = {}
    local f = csv.open(strFilename, taLoadParams)

    if f ~= nil then
      for fields in f:lines() do
        taRanges[fields.gene] = { min=tonumber(fields.min), max=tonumber(fields.max) }
      end
    end

    return taRanges
  end

    function CData:__init(strDir, oDepGraph, dMinDist, isNoise, strFoldFilename, nStratifiedRowId)
      self.taData = self:priLoad(strDir, oDepGraph, dMinDist, isNoise, strFoldFilename, nStratifiedRowId)
      self.taGERanges = loadGERanges(strDir)
      self.oDepGraph = oDepGraph
      self.strDir = strDir
      self.strTestActualFilename = string.format("%s/folds/actual_%s.csv", strDir, strFoldFilename:sub(1, -5))
      self.strFoldFilename = strFoldFilename
    end
    
    function CData:savePred(tePred, strPrefix)
      strPrefix = strPrefix or ""
      self.strTestPredFilename = string.format("%s/folds/%spred_%s.csv", self.strDir, strPrefix, self.strFoldFilename:sub(1, -5))
      local tePredReordered = data.priGetDataOrigOrder(tePred, self.taNonTFGenes, self.oDepGraph:getNonTFOrders())
      myUtil.saveTensorAndHeaderToCsvFile(tePredReordered, self.taNonTFGenes,  self.strTestPredFilename)
      print("saved preds to " .. self.strTestPredFilename)
    end
    
    function CData:saveActual()
      local teTargetReordered = data.priGetDataOrigOrder(self.taData.target, self.taNonTFGenes, self.oDepGraph:getNonTFOrders())
      myUtil.saveTensorAndHeaderToCsvFile(teTargetReordered, self.taNonTFGenes, self.strTestActualFilename)
      print("saved actuals to " .. self.strTestActualFilename)
    end
    
  
  --[[
  function data.savePred(taData)
    myUtil.saveTensorAndHeaderToCsvFile(teData, taHeader, strFilename)
  end--]]
  
    function CData:getMinMaxNonTFs()
      local taNonTFOrders = self.oDepGraph:getNonTFOrders()
      local nNonTFs = self.oDepGraph:getNumNonTFs()
      local taMinMax = {teMins = torch.zeros(nNonTFs), teMaxs = torch.ones(nNonTFs)}

      if self.taGERanges ~= nil then
        for strGene, idx in pairs(taNonTFOrders) do
          taMinMax.teMins[idx] = self.taGERanges[strGene].min
          taMinMax.teMaxs[idx] = self.taGERanges[strGene].max
        end
      end

      return taMinMax
    end
  
  return data
  
end
