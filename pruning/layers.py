import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import to_var
from approximate import *
#from main import b_size
en_approx = 1
from parameters import *


class TailoredLayer(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(TailoredLayer, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x, actions=None):
        '''
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        '''

        #self.weight.data = self.weight.org
        #weight_FL =
        #[128,128.32.32]
        #x.abs().mean(-1).mean(-1)
        #template = torch.cuda.FloatTensor([1.0,2.0,3.0,4.0])

        if  isinstance(actions,torch.cuda.FloatTensor):
            sorted, indices = torch.sort(x.abs().mean(-1).mean(-1),dim=1,descending=False)
            #actions[0]=torch.cuda.FloatTensor([1,2,3,4])
            input_size = x.size()
            actions = actions.unsqueeze(-1).expand(input_size[0], actions.size()[1], int(input_size[1]/actions.size()[1])).contiguous().view(input_size[0],input_size[1])

            input_FL=torch.ones(actions.size()).cuda()
            for i in range(input_size[0]):
                input_FL[i, indices[i]] = actions[i]


            #input_FL=torch.gather(actions, dim=1, index=indices)
            input_FL=input_FL.unsqueeze(-1).unsqueeze(-1).expand(input_size[0],input_size[1],input_size[2],input_size[3])
            x = quant(x, input_FL)

            #actions.unsqueeze(1).expand(4, 32).contiguous().view(128, 1)

        #channel_FL = torch.zeros([x.size()[0], x.size()[1]]).cuda()
        #channel_FL[:,indices[:,0:int(x.size()[1] / 4)]]

        #torch.gather(input, dim=1, actions)
        #out[i][j][k] = input[i][index[i][j][k]][k]

        #channel_FL[indices[0:int(x.size()[1]/4)]] = actions[0]
        #channel_FL[indices[int(x.size()[1]/4)+1:int(x.size()[1]/2)]] = actions[1]
        #channel_FL[indices[int(x.size()[1] / 2) + 1 + 1:int(x.size()[1]*3 / 4)]] = actions[2]
        #channel_FL[indices[int(x.size()[1]*3 / 4) + 1:int(x.size()[1])]] = actions[3]
        #sorted, indices = torch.sort(x)
        #input_FL=channel_FL.unsqueeze(0).expand(x.size()[0],x.size()[1])
        #input_FL=input_FL.unsqueeze(-1).unsqueeze(-1).expand(x.size()[0],x.size()[1],x.size()[2],x.size()[3])

        #weight_size =  self.weight.size()
        #weight_FL = channel_FL.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(weight_size[0],weight_size[1],weight_size[2],weight_size[3])
        #shapes = actions.repeat(32,1).reshape(128,1)


        #temp_weight = self.weight
        #temp_weight = quant(temp_weight, weight_FL.float())






        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



