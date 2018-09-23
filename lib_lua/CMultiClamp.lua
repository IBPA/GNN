CMultiClamp = torch.class('nn.CMultiClamp', 'nn.Module')

function CMultiClamp:__init(teMins, teMaxs)
  self.teMins = teMins
  self.teMaxs = teMaxs
end

function CMultiClamp:updateOutput(input)
  self.output = input:clone()
  for i=1, input:size(2) do
    self.output:select(2, i):clamp(self.teMins[i], self.teMaxs[i])
  end
  
  return self.output
end