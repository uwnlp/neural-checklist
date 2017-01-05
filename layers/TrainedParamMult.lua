require('torch')

local TrainedParamMult, parent = torch.class('TrainedParamMult', 'nn.Module')

function TrainedParamMult:__init(inputSize, outputSize)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.batchWeight = torch.Tensor(1, outputSize, inputSize)
   self.batchGradWeight = torch.Tensor(1, outputSize, inputSize)
   self.sumBatchGradWeight = torch.Tensor(1, outputSize, inputSize)
   self:reset(0.35)
end

function TrainedParamMult:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
   else
      self.weight:uniform(-stdv, stdv)
   end

   return self
end

function TrainedParamMult:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      self.output:copy(self.bias)
      self.output:addmv(1, self.weight, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.output:addmm(0, self.output, 1, input, self.weight:t())
   elseif input:dim() == 3 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1), input:size(3))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      nElement = self.batchWeight:nElement()
      self.batchWeight:resize(nframe, self.weight:size(1), self.weight:size(2))
      if self.batchWeight:nElement() ~= nElement then
         self.batchWeight:zero()
      end
      self.batchWeight:repeatTensor(self.weight, nframe, 1, 1)
      self.output:baddbmm(0, self.output, 1, self.batchWeight, input)
--[[--
      for i=1,nframe do
         self.output[i]:addmm(0, self.output[i], 1, self.weight, input[i])
      end
--]]--
   else
      error('input must be vector or matrix')
   end
--   print(input)
--   print(self.output)
--   print(self.weight)
   return self.output
end

function TrainedParamMult:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.weight)
      elseif input:dim() == 3 then
         local nframe = input:size(1)
         self.batchWeight = torch.repeatTensor(self.weight, nframe, 1, 1)
         self.gradInput:baddbmm(0, 1, self.batchWeight:transpose(2,3), gradOutput)
--[[--
         for i=1,nframe do
            self.gradInput[i]:addmm(0, 1, self.weight:t(), gradOutput[i])
         end
--]]--
      end

      return self.gradInput
   end
end

function TrainedParamMult:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if input:dim() == 1 then
      self.gradWeight:addr(scale, gradOutput, input)
   elseif input:dim() == 2 then
      self.gradWeight:addmm(scale, gradOutput:t(), input)
   elseif input:dim() == 3 then
      local nframe = input:size(1)
      local nElement = self.batchGradWeight:nElement()
      self.batchGradWeight:resize(nframe, self.weight:size(1), self.weight:size(2))
      if self.batchGradWeight:nElement() ~= nElement then
         self.batchGradWeight:zero()
      end
      self.batchGradWeight:baddbmm(scale, gradOutput, input:transpose(2,3))
      self.sumBatchGradWeight:sum(self.batchGradWeight, 1)
      self.gradWeight = self.sumBatchGradWeight[1]
--[[--
      for i=1,nframe do
         self.gradWeight:addmm(scale, gradOutput[i], input[i]:t())
      end
--]]--
   end
end

-- we do not need to accumulate parameters when sharing
TrainedParamMult.sharedAccUpdateGradParameters = TrainedParamMult.accUpdateGradParameters


function TrainedParamMult:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end
