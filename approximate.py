import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np




class Quantize(Function):

    @staticmethod
    def forward(cxt, input, FL): #FL means fractions bit length, IL we set it to 1

        x = input #.clone()
        sign_bits = torch.sign(x.clone())
        x = torch.abs(x.clone())

        y = x * torch.pow(torch.cuda.FloatTensor([2]), FL)
        compensate = ((y-torch.floor(y))>=0.5).float()
        y = y + compensate
        y = y.floor()
        overflow = (y >= torch.pow(torch.cuda.FloatTensor([2]),FL+1) -1)
        underflow = (y < 1)

        y[overflow] = torch.pow(torch.cuda.FloatTensor([2]), FL+1)[overflow] -1
        y[underflow] = 0

        y = y / torch.pow((torch.cuda.FloatTensor([2])), FL)
        y = y * sign_bits

        return y

    '''
    @staticmethod
    def forward(cxt, input, FL): #FL means fractions bit length, IL we set it to 1

        x = input
        sign_bits = torch.sign(x)
        #x = torch.abs(x)
        x.abs_()

        y = x * torch.pow(torch.cuda.FloatTensor([2]), FL)
        compensate = ((y-torch.floor(y))>=0.5).float()
        y = y + compensate
        y = y.floor()
        overflow = (y >= torch.pow(torch.cuda.FloatTensor([2]),FL+1) -1)
        underflow = (y < 1)

        y[overflow] = torch.pow(torch.cuda.FloatTensor([2]), FL+1)[overflow] -1
        y[underflow] = 0

        y = y / torch.pow((torch.cuda.FloatTensor([2])), FL)
        y = y * sign_bits

        return y
    
    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output
        return grad_input,None,None,None,None
    '''
# aliases
quant = Quantize.apply