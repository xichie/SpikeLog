import sys
sys.path.append('/mnt/LogADEmpirical/logadempirical')

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepod.core.base_networks import LinearBlock
import numpy as np
from torch.autograd import Variable
import snntorch as snn
from snntorch import spikegen, surrogate
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


class SpikeNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_out) -> None:
        super(SpikeNet, self).__init__()
        # spike_grad = surrogate.sigmoid() 
        self.hidden = num_hidden
        # randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)
        
        # Initialize layers 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta=beta_in, threshold=thr_in, linear_features=num_hidden, learn_beta=True, learn_threshold=True)
        
        # randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(64)
        thr_hidden = torch.rand(64)
        # Initialize layers 2
        self.fc2 = nn.Linear(num_hidden, 64)
        self.lif2 = snn.RLeaky(beta=beta_hidden, threshold=thr_hidden, linear_features=64, learn_beta=True, learn_threshold=True)
        
        # randomly initialize decay rate for output neuron
        beta_out = torch.rand(num_out)
        # Initialize layers 3
        self.fc_out = torch.nn.Linear(in_features=64, out_features=num_out)
        self.li_out = snn.RLeaky(beta=beta_out, threshold=0.1, linear_features=num_out, learn_beta=True, reset_mechanism="none")
        
        
    def forward(self, x):
        num_steps = x.size()[0]
        # Initialize hidden states at t=0
        spk1, mem1 = self.lif1.init_rleaky()
        spk2, mem2 = self.lif2.init_rleaky()
        spk3, mem3 = self.li_out.init_rleaky()

        # Record the final layer
        mem3_rec = []
        spk3_rec = []

        for step in range(num_steps):
            x_timestep = x[step, :, :]
            cur1 = self.fc1(x_timestep)
            spk1, mem1 = self.lif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, spk2, mem2)
            cur_out = self.fc_out(spk2)
            spk3, mem3 = self.li_out(cur_out, spk3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec, dim=0)
    
    def forward_pass(self, x):
        num_steps = x.size()[0]
        # Initialize hidden states at t=0
        spk1, mem1 = self.lif1.init_rleaky()
        spk2, mem2 = self.lif2.init_rleaky()
        spk3, mem3 = self.li_out.init_rleaky()

        # Record the final layer
        mem1_rec = []
        spk1_rec = []
        mem2_rec = []
        spk2_rec = []
        mem3_rec = []
        spk3_rec = []

        for step in range(num_steps):
            x_timestep = x[step, :, :]
            cur1 = self.fc1(x_timestep)
            spk1, mem1 = self.lif1(cur1, spk1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, spk2, mem2)
            cur_out = self.fc_out(spk2)
            spk3, mem3 = self.li_out(cur_out, spk3, mem3)
                      
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(spk3_rec, dim=0), torch.stack(mem1_rec, dim=0), torch.stack(mem2_rec, dim=0), torch.stack(mem3_rec, dim=0)

class DualInputNet_Spike(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_out) -> None:
        super().__init__()
        self.spike_rnn = SpikeNet(num_inputs, num_hidden, num_out)
        
        self.rnn = nn.LSTM(input_size=num_out, hidden_size=num_out)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=num_out*2, out_features=1),
        )

    def forward(self, x1, x2):
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        # x1 = spikegen.delta(x1, threshold=1.0, off_spike=True)
        # x2 = spikegen.delta(x2, threshold=1.0, off_spike=True)
        x1 = self.spike_rnn(x1) # (seq_len, bs, num_out)
        x2 = self.spike_rnn(x2)
        x1, (_,_) = self.rnn(x1)
        x2, (_,_) = self.rnn(x2)
        
        out = self.output_layer(torch.concat((x1[-1], x2[-1]), 1))
        
        return out