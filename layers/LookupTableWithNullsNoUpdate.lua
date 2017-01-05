require("nn")
local LookupTableWithNullsNoUpdate, parent = torch.class('LookupTableWithNullsNoUpdate', 'nn.Module')

LookupTableWithNullsNoUpdate.__version = 1

function LookupTableWithNullsNoUpdate:__init(nIndex, nOutput)
   parent.__init(self)

   self.weight = torch.Tensor(nIndex + 1, nOutput)
   self.gradWeight = torch.Tensor(nIndex + 1, nOutput):zero()

   self:reset()
   self:resetNullWeight()
end

function LookupTableWithNullsNoUpdate:resetNullWeight()
   self.weight:select(1,1):zero()
end

function LookupTableWithNullsNoUpdate:backCompatibility()
    self._count = self._count or torch.IntTensor()
    self._input = self._input or torch.LongTensor()
    self._gradOutput = self._gradOutput or torch.CudaTensor()

    if self.shouldScaleGradByFreq == nil then
        self.shouldScaleGradByFreq = false
    end
end

function LookupTableWithNullsNoUpdate:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function LookupTableWithNullsNoUpdate:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function LookupTableWithNullsNoUpdate:reset(stdv)
   stdv = stdv or 1
   self.weight:normal(0, stdv)
end

function LookupTableWithNullsNoUpdate:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end
function LookupTableWithNullsNoUpdate:makeGradOutputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._gradOutput) then
      self.copiedGradOutput = true
      self._gradOutput:resize(input:size()):copy(input)
      return self._gradOutput
   end
   self.copiedGradOutput = false
   return input
end

function LookupTableWithNullsNoUpdate:updateOutput(input)
   self:backCompatibility()
   input = self:makeInputContiguous(input)
   input:add(1)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   elseif input:dim() == 3 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), input:size(3), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   input:add(-1)
   return self.output
end

function LookupTableWithNullsNoUpdate:accGradParameters(input, gradOutput, scale)
--[[--
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() == 3 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end
   input:add(1)
   gradOutput = self:makeGradOutputContiguous(gradOutput)
   self.gradWeight.nn.LookupTable_accGradParameters(self, input, gradOutput, scale)
   input:add(-1)
--]]--
end

function LookupTableWithNullsNoUpdate:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
      self._count = self.weight.new()
      self._input = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

-- we do not need to accumulate parameters when sharing
LookupTableWithNullsNoUpdate.sharedAccUpdateGradParameters = LookupTableWithNullsNoUpdate.accUpdateGradParameters

