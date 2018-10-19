import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict
import torch
from utee import misc
print = misc.logger.info
from utee import rnn_model
from pruning.layers import  TailoredLayer
from parameters import *
import math

model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}
enable_rnn = 1
quant_output = 1

def convert_to_actions(action_scores):
    global steps_done
    actions_taken = action_scores.max(dim=1)[1]

    sample = torch.rand(1)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
              math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #print(eps_threshold)
    if sample < eps_threshold:
        #actions_taken = torch.randint(low=0, high=N_ACTIONS, size=(actions_taken.size()[0],)).cuda().long()
        actions_taken = torch.randint(low=0, high=N_ACTIONS, size=(actions_taken.size()[0],)).cuda().long()

    #actions_taken = torch.randint(low=0, high=1, size=(actions_taken.size()[0],)).cuda().long()
    print(actions_taken)
    return actions_taken


class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        #self.features._modules['3'] = TailoredLayer(128, 128, kernel_size=3, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x, rnn_ins=None, memory=None, target=None):
        #x = self.features(x)
        #x = x.view(x.size(0), -1)
        #x = self.classifier(x)
        #features_len = self.features._modules.__len__()
        #classifier_len = self.classifier._modules.__len__()
        #isinstance(self.features._modules['0'], nn.Conv2d)

        # global pooling to get x
        # X*W1 = FIXED
        # [X,C] concat CELL units C to get RNN inputs X
        # [H,C]=RNN(X, H)
        # REWARDS = FC(H)
        y = self.features(x)
        y = y.view(y.size(0), -1)
        y = self.classifier(y)


        tt=0
        normalizer= 0.0001*0.4
        coef_loss = 2
        rnn_ins.hidden = rnn_ins.init_hidden(x.size()[0])
        x = self.features._modules['0'](x)
        x = self.features._modules['1'](x)
        x = self.features._modules['2'](x)
        #state1 = [torch.mean(torch.mean(torch.abs(x),dim=3),dim=2) , rnn_ins.hidden]
        i=0
        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,1)
            actions = convert_to_actions(scores)
            p_q_action = torch.index_select(mappings,0,actions) #mappings[actions.item()]
            n_ifms = x.size()[1]
            x = self.features._modules['3'](x, p_q_action)
        else:
            x = self.features._modules['3'](x)
        x = self.features._modules['4'](x)
        x = self.features._modules['5'](x)
        x = self.features._modules['6'](x)

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                    torch.mean(torch.mean(torch.abs(x),dim=3),dim=2), rnn_ins.hidden, i+2, -1*normalizer*n_ifms*p_q_action.sum(dim=1))
            tt += -1*normalizer*n_ifms * p_q_action.sum().item()
        i+=1

        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,2)
            actions = convert_to_actions(scores)
            p_q_action = torch.index_select(mappings,0,actions)
            n_ifms = x.size()[1]
            x = self.features._modules['7'](x, p_q_action)
        else:
            x = self.features._modules['7'](x)
        x = self.features._modules['8'](x)
        x = self.features._modules['9'](x)

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                        torch.mean(torch.mean(torch.abs(x),dim=3),dim=2), rnn_ins.hidden, i+2, -1*normalizer*n_ifms*p_q_action.sum(dim=1))
            tt += -1*normalizer*n_ifms * p_q_action.sum().item()
        i+=1

        # state3 = [torch.mean(torch.mean(torch.abs(x),dim=3),dim=2) , rnn_ins.hidden ]
        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,3)
            actions = convert_to_actions(scores)
            p_q_action = torch.index_select(mappings,0,actions)
            n_ifms = x.size()[1]
            x = self.features._modules['10'](x, p_q_action)
        else:
            x = self.features._modules['10'](x)
        x = self.features._modules['11'](x)
        x = self.features._modules['12'](x)
        x = self.features._modules['13'](x)

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                        torch.mean(torch.mean(torch.abs(x),dim=3),dim=2), rnn_ins.hidden, i+2, -1*normalizer*n_ifms*p_q_action.sum(dim=1))
            tt += -1*normalizer*n_ifms * p_q_action.sum().item()
        i += 1


        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,4)
            actions = convert_to_actions(scores)
            p_q_action = torch.index_select(mappings,0,actions)
            n_ifms = x.size()[1]
            x = self.features._modules['14'](x, p_q_action)
        else:
            x = self.features._modules['14'](x)
        x = self.features._modules['15'](x)
        x = self.features._modules['16'](x)

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                        torch.mean(torch.mean(torch.abs(x),dim=3),dim=2), rnn_ins.hidden, i+2, -1*normalizer*n_ifms*p_q_action.sum(dim=1))
            tt += -1*normalizer*n_ifms * p_q_action.sum().item()
        i += 1


        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,5)
            actions = convert_to_actions(scores)
            n_ifms = x.size()[1]
            p_q_action = torch.index_select(mappings,0,actions)
            x = self.features._modules['17'](x, p_q_action)
        else:
            x = self.features._modules['17'](x)
        x = self.features._modules['18'](x)
        x = self.features._modules['19'](x)
        x = self.features._modules['20'](x)

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                        torch.mean(torch.mean(torch.abs(x),dim=3),dim=2), rnn_ins.hidden, i+2, -1*normalizer*n_ifms*p_q_action.sum(dim=1))
            tt += -1*normalizer*n_ifms * p_q_action.sum().item()
        i += 1


        if enable_rnn:
            prev_input = torch.mean(torch.mean(torch.abs(x), dim=3), dim=2)
            prev_hidden = rnn_ins.hidden
            input = torch.mean(torch.mean(torch.abs(x),dim=3),dim=2)
            scores = rnn_ins(input,6)
            actions = convert_to_actions(scores)
            n_ifms = x.size()[1]
            p_q_action = torch.index_select(mappings,0,actions)
            x = self.features._modules['21'](x, p_q_action)
        else:
            x = self.features._modules['21'](x)



        x = self.features._modules['22'](x)
        x = self.features._modules['23'](x)
        x = self.features._modules['24'](x)
        x = x.view(x.size(0), -1)
        x = self.classifier._modules['0'](x)

        criterion = nn.CrossEntropyLoss(reduce=False).cuda()
        #criterion = nn.MSELoss().cuda() #reduce=False
        loss = criterion(x, target).detach()

        loss2 = criterion(y, target).detach()
        mask = loss2 > 1.0
        relative_loss = loss - loss2
        #relative_loss[relative_loss<0.1]=0
        relative_loss=loss
        relative_loss[mask]=0
        #relative_loss[relative_loss>0.2]=0.2

        if enable_rnn:
            memory[i].push(prev_input, prev_hidden, i+1, actions, \
                        None, None, None,  -1*coef_loss*relative_loss + -1*normalizer*n_ifms*p_q_action.sum(dim=1))

            tt += -1*normalizer*n_ifms * p_q_action.sum().item()-1*coef_loss*relative_loss.sum().item()
        print( str(tt)+ '\n')
        print( -coef_loss*relative_loss.sum().item())
        return x, tt

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            if i == 0:
                conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            else:
                conv2d = TailoredLayer(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar100'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

if __name__ == '__main__':
    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    embed()
