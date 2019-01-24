--[[ Description: CDepGraph class loads graph from .dep and performs relevent functions to construct GNN nodes in appropriate order.
]]

CDepGraph = torch.class('CDepGraph')

function CDepGraph:__init(strDepFilename)
  self.strDepFilename = strDepFilename
  
  self.taAll = {}
  self.taOrder = {}
  self.nTFs = 0
  self.nSize = 0
  self.taNonTFs = {}
  
  local file = io.open(strDepFilename, "r")
  for strLine in file:lines() do
    local taSplit1 = strLine:split(':')
    local strGene = taSplit1[1]
    
    if #taSplit1 >1 then
      self.taAll[strGene] = taSplit1[2]:split(',')
      table.insert(self.taNonTFs, strGene)
    else
      self.nTFs = self.nTFs + 1
      self.taAll[strGene] = {}
    end
    
    self.nSize = self.nSize + 1
    self.taOrder[strGene] = self.nSize
  end
end

function CDepGraph:getNonTFs()
  --[[
  local taNonTFs = {}
  for strGene, taValue in pairs(self.taAll) do
    if #taValue > 0 then
      table.insert(taNonTFs, self.taOrder[strGene] - self.nTFs, strGene)
    end
  end --]]
  
  return self.taNonTFs
end

function CDepGraph:getNumNonTFs()
  return self.nSize - self.nTFs
end

function CDepGraph:getTFOrders()
  local taTFOrder = {}
  for strGene, nOrder in pairs(self.taOrder) do
    if #self.taAll[strGene] == 0 then
      taTFOrder[strGene] = nOrder
    end
  end
  
  return taTFOrder
end

function CDepGraph:getNonTFOrders()
   local taNonTFOrder = {}
  for strGene, nOrder in pairs(self.taOrder) do
    if #self.taAll[strGene] > 0 then
      taNonTFOrder[strGene] = nOrder - self.nTFs
    end
  end
  
  return taNonTFOrder
end

function CDepGraph:getDepIds(strGene)
  local taDepIds = {}
  local taDepNames = {}
  local taDeps = self.taAll[strGene]
  for __, strDep in pairs(taDeps) do
    table.insert(taDepIds, self.taOrder[strDep])
    table.insert(taDepNames, strDep)
  end
  
  return taDepIds, taDepNames
end



function CDepGraph:getnBefore(strGene)
  return self.taOrder[strGene] - 1
end
