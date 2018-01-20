from __future__ import print_function

import os
import argparse
import torch.legacy.nn as lnn

from torch.utils.serialization import load_lua

from header import *

def copy_param(m, n):
    if m.weight is not None: n.weight.data.copy_(m.weight, broadcast=False)
    if hasattr(m, 'bias') and m.bias is not None: n.bias.data.copy_(m.bias, broadcast=False)
    if hasattr(n, 'running_mean'): n.running_mean.copy_(m.running_mean, broadcast=False)
    if hasattr(n, 'running_var'): n.running_var.copy_(m.running_var, broadcast=False)


def add_submodule(seq, m):
    seq.add_module(str(len(seq._modules)), m)


def lua_recursive_model(module, seq):
    for m in module.modules:
        name = type(m).__name__
        real = m
        n = None
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.', '')
            m = m._obj
        if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM' or name == 'SpatialDilatedConvolution':
            if not hasattr(m, 'dilationH') or m.dilationH is None: m.dilationH, m.dilationW = (1, 1)
            if not hasattr(m, 'groups') or m.groups is None: m.groups = 1
            n = nn.Conv2d(m.nInputPlane, m.nOutputPlane, (m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW), (m.dilationH, m.dilationW), m.groups,
                          bias=(m.bias is not None))
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'SpatialBatchNormalization':
            n = nn.BatchNorm2d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'VolumetricBatchNormalization':
            n = nn.BatchNorm3d(m.running_mean.size(0), m.eps, m.momentum, m.affine)
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'ReLU':
            n = nn.ReLU()
            add_submodule(seq, n)
        elif name == 'PReLU':
            n = nn.PReLU(1 if m.nOutputPlane == 0 else m.nOutputPlane)
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'Padding':
            n = Padding(m.dim, m.pad, value=0, index=0, nInputDim=0)
            add_submodule(seq, n)
        elif name == 'Sigmoid':
            n = nn.Sigmoid()
            add_submodule(seq, n)
        elif name == 'SpatialMaxPooling':
            n = StatefulMaxPool2d((m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW), ceil_mode=m.ceil_mode)
            add_submodule(seq, n)
        elif name == 'SpatialMaxUnpooling':
            n = StatefulMaxUnpool2d()
            add_submodule(seq, n)
        elif name == 'SpatialAveragePooling':
            n = nn.AvgPool2d((m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW), ceil_mode=m.ceil_mode)
            add_submodule(seq, n)
        elif name == 'SpatialUpSamplingNearest':
            n = nn.UpsamplingNearest2d(scale_factor=m.scale_factor)
            add_submodule(seq, n)
        elif name == 'View':
            n = Lambda(lambda x: x.view(x.size(0), -1))
            add_submodule(seq, n)
        elif name == 'Reshape':
            n = Lambda(lambda x: x.view(x.size(0), -1))
            add_submodule(seq, n)
        elif name == 'Linear':
            # Linear in pytorch only accept 2D input
            n1 = Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x)
            n2 = nn.Linear(m.weight.size(1), m.weight.size(0), bias=(m.bias is not None))
            copy_param(m, n2)
            n = nn.Sequential(n1, n2)
            add_submodule(seq, n)
        elif name == 'Dropout':
            m.inplace = False
            n = Dropout(m.p)
            add_submodule(seq, n)
        elif name == 'SpatialDropout':
            m.inplace = False
            n = Dropout2d(m.p)
            add_submodule(seq, n)
        elif name == 'SoftMax':
            n = nn.Softmax()
            add_submodule(seq, n)
        elif name == 'Identity':
            n = Lambda(lambda x: x)  # do nothing
            add_submodule(seq, n)
        elif name == 'SpatialFullConvolution':
            n = nn.ConvTranspose2d(m.nInputPlane, m.nOutputPlane, (m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW),
                                   (m.adjH, m.adjW))
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'VolumetricFullConvolution':
            n = nn.ConvTranspose3d(m.nInputPlane, m.nOutputPlane, (m.kT, m.kH, m.kW), (m.dT, m.dH, m.dW),
                                   (m.padT, m.padH, m.padW), (m.adjT, m.adjH, m.adjW), m.groups)
            copy_param(m, n)
            add_submodule(seq, n)
        elif name == 'SpatialReplicationPadding':
            n = nn.ReplicationPad2d((m.pad_l, m.pad_r, m.pad_t, m.pad_b))
            add_submodule(seq, n)
        elif name == 'SpatialReflectionPadding':
            n = nn.ReflectionPad2d((m.pad_l, m.pad_r, m.pad_t, m.pad_b))
            add_submodule(seq, n)
        elif name == 'Copy':
            n = Lambda(lambda x: x)  # do nothing
            add_submodule(seq, n)
        elif name == 'Narrow':
            n = Lambda(lambda x, a=(m.dimension, m.index, m.length): x.narrow(*a))
            add_submodule(seq, n)
        elif name == 'SpatialCrossMapLRN':
            lrn = lnn.SpatialCrossMapLRN(m.size, m.alpha, m.beta, m.k)
            n = Lambda(lambda x, lrn=lrn: Variable(lrn.forward(x.data)))
            add_submodule(seq, n)
        elif name == 'Sequential':
            n = nn.Sequential()
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'ConcatTable':  # output is list
            n = LambdaMap(lambda x: x)
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'CAddTable':  # input is list
            n = LambdaReduce(lambda x, y: x + y)
            add_submodule(seq, n)
        elif name == 'JoinTable':
            n = LambdaReduce(lambda x, y: torch.cat((x, y), m.dimension))
            add_submodule(seq, n)
        elif name == 'Concat':
            n = LambdaReduce(lambda x, y: torch.cat((x, y), m.dimension))
            lua_recursive_model(m, n)
            add_submodule(seq, n)
        elif name == 'TorchObject':
            print('Not Implement', name, real._typename)
        else:
            print('Not Implement', name)


pooling_instances = 0
def lua_recursive_source(module):
    global pooling_instances
    s = []
    for m in module.modules:
        name = type(m).__name__
        real = m
        if name == 'TorchObject':
            name = m._typename.replace('cudnn.', '')
            m = m._obj

        if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM' or name == 'SpatialDilatedConvolution':
            if not hasattr(m, 'dilationH') or m.dilationH is None: m.dilationH, m.dilationW = (1, 1)
            if not hasattr(m, 'groups') or m.groups is None: m.groups = 1
            s += ['nn.Conv2d({}, {}, {}, {}, {}, {}, {}, bias={}), # Conv2d'.format(m.nInputPlane,
                                                                           m.nOutputPlane, (m.kH, m.kW), (m.dH, m.dW),
                                                                           (m.padH, m.padW), (m.dilationW, m.dilationH), m.groups,
                                                                           m.bias is not None)]
        elif name == 'SpatialBatchNormalization':
            s += [
                'nn.BatchNorm2d({}, {}, {}, {}), # BatchNorm2d'.format(m.running_mean.size(0), m.eps, m.momentum, m.affine)]
        elif name == 'VolumetricBatchNormalization':
            s += [
                'nn.BatchNorm3d({}, {}, {}, {}), # BatchNorm3d'.format(m.running_mean.size(0), m.eps, m.momentum, m.affine)]
        elif name == 'ReLU':
            s += ['nn.ReLU()']
        elif name == 'PReLU':
            s += ['nn.PReLU({})'.format(1 if m.nOutputPlane == 0 else m.nOutputPlane)]
        elif name == 'Padding':
            s += ['Padding({}, {}, {}, {}, {})'.format(m.dim, m.pad, m.value, m.index, m.nInputDim)]
        elif name == 'Sigmoid':
            s += ['nn.Sigmoid()']
        elif name == 'SpatialMaxPooling':
            pooling_instances += 1
            s += ['StatefulMaxPool2d({}, {}, {}, ceil_mode={}), #SpatialMaxPooling {}'.format((m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW),
                                                                          m.ceil_mode, pooling_instances)]
        elif name == 'SpatialMaxUnpooling':
            print('Warning: using FIFO list for MaxUnpooling (symmetric decoding)')
            s += ['StatefulMaxUnpool2d(), #SpatialMaxUnpooling {}'.format(pooling_instances)]
            pooling_instances -= 1
        elif name == 'SpatialAveragePooling':
            s += ['nn.AvgPool2d({}, {}, {}, ceil_mode={}), #AvgPool2d'.format((m.kH, m.kW), (m.dH, m.dW), (m.padH, m.padW),
                                                                          m.ceil_mode)]
        elif name == 'SpatialUpSamplingNearest':
            s += ['nn.UpsamplingNearest2d(scale_factor={})'.format(m.scale_factor)]
        elif name == 'View':
            s += ['Lambda(lambda x: x.view(x.size(0), -1)), # View']
        elif name == 'Reshape':
            s += ['Lambda(lambda x: x.view(x.size(0), -1)), # Reshape']
        elif name == 'Linear':
            s1 = 'Lambda(lambda x: x.view(1, -1) if 1==len(x.size()) else x )'
            s2 = 'nn.Linear({}, {}, bias={})'.format(m.weight.size(1), m.weight.size(0), (m.bias is not None))
            s += ['nn.Sequential({}, {}), #Linear'.format(s1, s2)]
        elif name == 'Dropout':
            s += ['Dropout({})'.format(m.p)]
        elif name == 'SpatialDropout':
            s += ['Dropout2d({})'.format(m.p)]
        elif name == 'SoftMax':
            s += ['nn.Softmax()']
        elif name == 'Identity':
            s += ['Lambda(lambda x: x), # Identity']
        elif name == 'SpatialFullConvolution':
            s += ['nn.ConvTranspose2d({}, {}, {}, {}, {}, {})'.format(m.nInputPlane,
                                                                 m.nOutputPlane, (m.kH, m.kW), (m.dH, m.dW),
                                                                 (m.padH, m.padW), (m.adjH, m.adjW))]
        elif name == 'VolumetricFullConvolution':
            s += ['nn.ConvTranspose3d({}, {}, {}, {}, {}, {}, {})'.format(m.nInputPlane,
                                                                    m.nOutputPlane, (m.kT, m.kH, m.kW),
                                                                    (m.dT, m.dH, m.dW), (m.padT, m.padH, m.padW),
                                                                    (m.adjT, m.adjH, m.adjW), m.groups)]
        elif name == 'SpatialReplicationPadding':
            s += ['nn.ReplicationPad2d({})'.format((m.pad_l, m.pad_r, m.pad_t, m.pad_b))]
        elif name == 'SpatialReflectionPadding':
            s += ['nn.ReflectionPad2d({})'.format((m.pad_l, m.pad_r, m.pad_t, m.pad_b))]
        elif name == 'Copy':
            s += ['Lambda(lambda x: x), # Copy']
        elif name == 'Narrow':
            s += ['Lambda(lambda x, a={}: x.narrow(*a))'.format((m.dimension, m.index, m.length))]
        elif name == 'SpatialCrossMapLRN':
            lrn = 'lnn.SpatialCrossMapLRN(*{})'.format((m.size, m.alpha, m.beta, m.k))
            s += ['Lambda(lambda x,lrn={}: Variable(lrn.forward(x.data)))'.format(lrn)]

        elif name == 'Sequential':
            s += ['nn.Sequential( # Sequential']
            s += lua_recursive_source(m)
            s += [')']
        elif name == 'ConcatTable':
            s += ['LambdaMap(lambda x: x, # ConcatTable']
            s += lua_recursive_source(m)
            s += [')']
        elif name == 'JoinTable':
            s += ['LambdaReduce(lambda x, y: torch.cat((x, y), {}))'.format(m.dimension)]
        elif name == 'CAddTable':
            s += ['LambdaReduce(lambda x,y: x+y), # CAddTable']
        elif name == 'Concat':
            s += ['LambdaReduce(lambda x,y: torch.cat((x,y), {}), # Concat'.format(m.dimension)]
            s += lua_recursive_source(m)
            s += [')']
        else:
            s += ['# ' + name + ' Not Implemented']
    s = map(lambda x: '    {}'.format(x), s)
    return s


def simplify_source(s):
    s = map(lambda x: x.replace(', (1, 1), (0, 0), (1, 1), 1, bias=True), # Conv2d', ')'), s)
    s = map(lambda x: x.replace(', (0, 0), 1, 1, bias=True), # Conv2d', ')'), s)
    s = map(lambda x: x.replace(', 1, 1, bias=True), # Conv2d', ')'), s)
    s = map(lambda x: x.replace(', bias=True), # Conv2d', ')'), s)
    s = map(lambda x: x.replace('), # Conv2d', ')'), s)
    s = map(lambda x: x.replace(', 1e-05, 0.1, True), # BatchNorm2d', ')'), s)
    s = map(lambda x: x.replace('), # BatchNorm2d', ')'), s)
    s = map(lambda x: x.replace(', (0, 0), ceil_mode=False), #MaxPool2d', ')'), s)
    s = map(lambda x: x.replace(', ceil_mode=False), #MaxPool2d', ')'), s)
    s = map(lambda x: x.replace('), #MaxPool2d', ')'), s)
    s = map(lambda x: x.replace(', (0, 0), ceil_mode=False), #AvgPool2d', ')'), s)
    s = map(lambda x: x.replace(', ceil_mode=False), #AvgPool2d', ')'), s)
    s = map(lambda x: x.replace(', bias=True)), #Linear', ')), # Linear'), s)
    s = map(lambda x: x.replace(')), #Linear', ')), # Linear'), s)

    s = map(lambda x: '{}, \n'.format(x), s)
    s = map(lambda x: x[4:], s)
    s = reduce(lambda x, y: x + y, s)
    return s


def torch_to_pytorch(t7_filename, varname=None):
    model = load_lua(t7_filename, unknown_classes=True)
    if type(model).__name__ == 'hashable_uniq_dict': model = model.model
    model.gradInput = None
    slist = lua_recursive_source(lnn.Sequential().add(model))
    s = simplify_source(slist)
    header = open("header.py").read()
    if varname is None:
        varname = os.path.splitext(os.path.basename(t7_filename))[0].replace('.', '_').replace('-', '_')
    outputname = os.path.join(os.path.dirname(t7_filename), varname)
    s = '{}\n\n{}, = {}'.format(header, varname, s[:-2])

    with open(outputname + '.py', "w") as pyfile:
        pyfile.write(s)

    n = nn.Sequential()
    lua_recursive_model(model, n)
    torch.save(n.state_dict(), outputname + '.pth')


parser = argparse.ArgumentParser(description='Convert torch t7 model to pytorch')
parser.add_argument('--model', '-m', type=str, required=True,
                    help='torch model file in t7 format')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='output file name prefix, xxx.py xxx.pth')
args = parser.parse_args()

torch_to_pytorch(args.model, args.output)
