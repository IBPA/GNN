--[[ Description: CParamCache class used for caching nn information (debugging purposes only).
]]
CParamCache = torch.class('CParamCache')

function CParamCache:__init()
  self.taCache = {}
  self.nFreqCache = 2
  self.nCount = 0
end

function CParamCache:update(teP, dErr)
  if dErr then
    local nD = teP:size(1)
    if self.taCache[nD] == nil then
      self.taCache[nD] = {}
    end
    
    table.insert(self.taCache[nD], {teP = teP:clone(), dErr = dErr})
  end
end

function CParamCache:getNew(nD)
  return (torch.rand(nD)*2 - 1)
end

function CParamCache:get(nD)
  local taCurrCache = self.taCache[nD]

  if taCurrCache ~= nil then
    self.nCount = self.nCount + 1
    
    if (self.nCount % self.nFreqCache == 0 ) then
      local idx = math.random(1, #taCurrCache)
      return taCurrCache[idx].teP:clone()
    end
    
  end
  
  return self:getNew(nD)
  
end
