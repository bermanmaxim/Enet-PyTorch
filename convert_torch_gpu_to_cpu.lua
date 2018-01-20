require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU torch net to CPU torch net.')
cmd:text()
cmd:text('Options')
cmd:argument('-model', 'GPU model checkpoint to convert')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

local gpumodel = torch.load(opt.model)

cpumodel = cudnn.convert(gpumodel, nn):float()

local savefile = 'cpu-' .. opt.model -- prepend "cpu-" to filename
torch.save(savefile, cpumodel)
print('saved ' .. savefile)
