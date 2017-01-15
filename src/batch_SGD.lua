local StochasticGradient = torch.class('batch_SGD')

function StochasticGradient:__init(module, criterion)
   self.learningRate = 0.01
   self.learningRateDecay = 0
   self.batchSize = 10
   self.maxIteration = 25
   self.shuffleIndices = true
   self.module = module
   self.criterion = criterion
   self.verbose = true
end

function StochasticGradient:train(dataset)
   local iteration = 1
   local currentLearningRate = self.learningRate
   local module = self.module
   local criterion = self.criterion

   local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
   if not self.shuffleIndices then
      for t = 1,dataset:size() do
         shuffledIndices[t] = t
      end
   end

   print("# batch_SGD: training")
   
   
   
   input_shape = {self.batchSize}
   if(type(dataset[1][1]) == "number") then
   	   table.insert(input_shape,1)
   else 
   	   local temp = dataset[1][1]:size()
	   for i=1,temp:size() do
	   		table.insert(input_shape,temp[i])
	   end
   end
   input_shape = torch.LongStorage(input_shape)
   
   
   target_shape = {self.batchSize}
   if(type(dataset[1][2]) == "number") then
   	   table.insert(target_shape,1)
   else 
   	   local temp = dataset[1][2]:size()
	   for i=1,temp:size() do
	   		table.insert(target_shape,temp[i])
	   end
   end
   target_shape = torch.LongStorage(target_shape)
   while true do
      local currentError = 0
      for t = 1,dataset:size(),self.batchSize do
      	local inputs = torch.Tensor(input_shape):cuda()
      	local targets = torch.Tensor(target_shape):cuda()
      	idx = t
      	for i = 1,self.batchSize do
      	 	local example = dataset[shuffledIndices[t]]
      		inputs[i] = example[1]
      		targets[i] = example[2]
      		idx = idx + 1
      		if(idx > dataset:size()) then
      			idx = 1
      		end
      	end
		 val = module:forward(inputs)
         currentError = currentError + criterion:forward(val, targets)

         module:updateGradInput(inputs, criterion:updateGradInput(module.output, targets))
         module:accUpdateGradParameters(inputs, criterion.gradInput, currentLearningRate)

         if self.hookExample then
            self.hookExample(self, example)
         end
      end

      currentError = currentError

      if self.hookIteration then
         self.hookIteration(self, iteration, currentError)
      end

      if self.verbose then
         print("# epoch = " .. iteration .." current error = " .. currentError)
      end
      iteration = iteration + 1
      currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
         print("# StochasticGradient: you have reached the maximum number of iterations")
         print("# training error = " .. currentError)
         break
      end
   end
end
