import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1)
import random


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class LSTMController(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, channel_conf, num_action, batch_size):
        super(LSTMController, self).__init__()
        self.hidden_dim = hidden_dim
        layers_list = []
        hidden_list = []
        for i, v in enumerate(channel_conf):
            in_channels=v
            layers_list += [nn.Linear(in_channels, embedding_dim).cuda()]

        for i, v in enumerate(channel_conf):
            hidden_list += [nn.Linear(hidden_dim, num_action).cuda()]
        hidden_list[-1] = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.layers = ListModule(*layers_list)
        self.hidden_action_array = ListModule(*hidden_list)
        #self.test_layer = nn.Linear(30,100)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim).cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2action = nn.Linear(hidden_dim, num_action).cuda()
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, batch_size, self.hidden_dim).cuda(),
                torch.zeros(1, batch_size, self.hidden_dim).cuda())

    def set_hidden(self, hidden):
        self.hidden = hidden

    def forward(self, input, idx):

        embeds = F.relu(self.layers[idx](input))   #self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(1, input.size()[0], -1), self.hidden)
        lstm_out = F.relu(lstm_out)
        action_space = self.hidden_action_array[idx](lstm_out.squeeze(0))
        #action_space = self.hidden2action(lstm_out.squeeze(0))
        if idx == 6:
            action_space = self.hidden2action(F.relu(action_space))
        action_scores = (action_space)# F.log_softmax(action_space, dim=1)


        return action_scores
