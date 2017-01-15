require 'torch'
require 'nn'
require 'nngraph'
require "cutorch"
require 'cunn'
require 'batch_SGD'
require 'optim'

local nninit = require 'nninit'
train_dir = '../toneclassifier/train/'
test_dir = '../toneclassifier/test_new/'



opt = {
	input_channel = 1,
	
	c1_channel = 2,
	c1_size = 10,

	p1_size = 4,
	p1_stride = 2,

	c2_channel = 4,
	c2_size = 10,

	p2_size = 4,
	p2_stride = 2,

	c3_channel = 8,
	c3_size = 10,

	p3_size = 4,
	p3_stride = 2,
	
	
	h1_size=2048,
	h2_size=2048,
	output_size=4
}
scale_l = 0
scale_r = 0.05
--Setup layers
net = nn.Sequential()

net:add(nn.SpatialConvolutionMM(opt.input_channel, opt.c1_channel,opt.c1_size, 1, 1, 1, (opt.c1_size-1) / 2, 0):init('weight',nninit.normal,scale_l,scale_r)) 

net:add(nn.SpatialMaxPooling(opt.p1_size,1,opt.p1_stride,1))

net:add(nn.SpatialConvolutionMM(opt.c1_channel, opt.c2_channel,opt.c2_size, 1, 1, 1, (opt.c2_size-1) / 2, 0):init('weight',nninit.normal,scale_l,scale_r)) 

net:add(nn.SpatialMaxPooling(opt.p2_size,1,opt.p2_stride,1))

net:add(nn.SpatialConvolutionMM(opt.c2_channel, opt.c3_channel,opt.c3_size, 1, 1, 1, (opt.c3_size-1) / 2, 0):init('weight',nninit.normal,scale_l,scale_r)) 

net:add(nn.SpatialMaxPooling(opt.p3_size,1,opt.p3_stride,1))


size = 8 * 1 * 9
net:add(nn.View(size))


net:add(nn.Linear(size,opt.h1_size):init('weight',nninit.normal,scale_l,scale_r))
net:add(nn.Sigmoid())
net:add(nn.Linear(opt.h1_size,opt.h2_size):init('weight',nninit.normal,scale_l,scale_r))
net:add(nn.Sigmoid())
net:add(nn.Linear(opt.h2_size,opt.output_size):init('weight',nninit.normal,scale_l,scale_r))

print('CNN :\n' .. net:__tostring());

-- Setup Criterions
criterion = nn.CrossEntropyCriterion()



-- Setup dataset

do
	local dataset = torch.class('Dataset')
	function dataset:__init(input,output,cuda)
		self.cuda = cuda
		self.input = input
		self.output = output + 1
	end
	function dataset:size()
		return self.input:size(1)
	end
	function dataset:__index__(v)
	    if type(v) == "number" then
	    if self.cuda == false then
				local tbl =  {
	            torch.reshape(self.input[v],1,1,self.input[v]:size(1)),
	            self.output[v]
	        	}
	        	return tbl, true
	    	else
				local tbl =  {
	            torch.reshape(self.input[v],1,1,self.input[v]:size(1)):cuda(),
	            self.output[v]
	        	}
	        	return tbl, true
	    	end
	        
	    else
	        return false
	    end
	end
end

function eval(dataset)
	local correct = 0
	for idx = 1,dataset:size() do
		cur = dataset[idx]
		groundtruth = cur[2]
		prediction = net:forward(cur[1])
		local confidences, indices = torch.sort(prediction, true)
		if groundtruth == indices[1] then
        	correct = correct + 1
    	end
	end
	return 100 * correct / dataset:size()
end

local csv2tensor = require('csv2tensor')
train_value_tensor = csv2tensor.load(train_dir .. 'datanew.csv') 
train_label_tensor = csv2tensor.load(train_dir .. 'labelnew.csv')

testnew_value_tensor = csv2tensor.load(test_dir .. 'datanew.csv') 
testnew_label_tensor = csv2tensor.load(test_dir .. 'labelnew.csv')

 
local trainset = Dataset(train_value_tensor,train_label_tensor,true)
local testnewset = Dataset(testnew_value_tensor,testnew_label_tensor,true)

-- Setup trainer


net=net:cuda()
criterion=criterion:cuda()


trainer = batch_SGD(net, criterion)
trainer.learningRate = 5e-3
trainer.maxIteration = 60

tic = sys.clock()
trainer:train(trainset)
toc = sys.clock()
print ("Total training time: " .. (toc - tic))
print (" train acc : " .. eval(trainset) .. " test_new acc : " .. eval(testnewset))




