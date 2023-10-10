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
from snntorch.functional import loss

class SpikeNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_out, batch_first=False) -> None:
        super(SpikeNet, self).__init__()
        self.batch_first = batch_first
        spike_grad = surrogate.sigmoid() 
        # spike_grad_lstm = surrogate.straight_through_estimator()

         # randomly initialize decay rate for output neuron
        beta_1 = torch.rand(num_hidden)
        beta_2 = torch.rand(num_hidden)
        beta_3 = torch.rand(num_out)
        
        # Initialize layers
        self.fc1 = torch.nn.Linear(in_features=num_inputs, out_features=num_hidden)
        self.lif1= snn.RLeaky(beta=beta_1, learn_threshold=True, linear_features=num_hidden, learn_beta=True)
        
        self.fc2 = torch.nn.Linear(in_features=num_hidden, out_features=num_hidden)
        self.lif2= snn.RLeaky(beta=beta_2, learn_threshold=True, linear_features=num_hidden, learn_beta=True)
        
        self.fc_out = torch.nn.Linear(in_features=num_hidden, out_features=num_out)
        self.li_out = snn.RLeaky(beta=beta_3, threshold=1.0, linear_features=num_out, learn_beta=True, learn_threshold=True, reset_mechanism="none")
        
    def forward(self, x):
        if self.batch_first:
            num_steps = x.size()[1]
        else:
            num_steps = x.size()[0]
        # Initialize hidden states and outputs at t=0
        spk1, mem1 = self.lif1.init_rleaky()
        spk2, mem2 = self.lif2.init_rleaky()
        spkout, memout = self.li_out.init_rleaky()
        # Record the final layer
        spkout_rec = []
        memout_rec = []
        if self.batch_first:
            for step in range(num_steps):
                x_timestep = x[:, step, :]
                cur_out = self.fc1(x_timestep)
                spk1, mem1 = self.lif1(cur_out, spk1, mem1)
                
                cur_out = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur_out, spk2, mem2)
                
                cur_out = self.fc_out(mem2)
                spkout, memout = self.li_out(cur_out, spkout,  memout)
                              
                spkout_rec.append(spkout)
                memout_rec.append(memout)
                
            return torch.stack(spkout_rec).permute(1, 0, 2), torch.stack(memout_rec).permute(1, 0, 2)
        else:
            for step in range(num_steps):
                x_timestep = x[step, :, :]
                cur_out = self.fc1(x_timestep)
                spk1, mem1 = self.lif1(cur_out, spk1, mem1)
                
                cur_out = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur_out, spk2, mem2)
                
                cur_out = self.fc_out(mem2)
                spkout, memout = self.li_out(cur_out, spkout,  memout)
                
                spkout_rec.append(spkout)
                memout_rec.append(memout)

            return torch.stack(spkout_rec), torch.stack(memout_rec)

class SpikeNet_LSTM(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_out, batch_first=True) -> None:
        super(SpikeNet_LSTM, self).__init__()
        # spike_grad = surrogate.sigmoid() 
        spike_grad_lstm = surrogate.straight_through_estimator()
        self.slstm1 = snn.SLSTM(input_size=num_inputs, hidden_size=num_hidden, \
                                reset_mechanism='subtract', spike_grad=spike_grad_lstm)
        # self.slstm2 = snn.SLSTM(input_size=num_hidden, hidden_size=num_hidden, \
        #                         reset_mechanism='subtract', spike_grad=spike_grad_lstm)
        # self.slstm3 = snn.SLSTM(input_size=num_hidden, hidden_size=num_hidden, \
        #                         reset_mechanism='subtract', spike_grad=spike_grad_lstm)
        self.slstm_out = snn.SLSTM(input_size=num_hidden, hidden_size=num_out, learn_threshold=True, \
                                    reset_mechanism='subtract', spike_grad=spike_grad_lstm)
        
        self.batch_first = batch_first
        
    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        syn1, mem1 = self.slstm1.init_slstm()
        # syn2, mem2 = self.slstm2.init_slstm()
        # syn3, mem3 = self.slstm3.init_slstm()
        synout, memout = self.slstm_out.init_slstm()
        # Record the final layer
        spkout_rec = []
        memout_rec = []
        if self.batch_first:
            num_steps = x.size()[1]
            for step in range(num_steps):
                x_timestep = x[:,step]
                spk1, syn1, mem1, = self.slstm1(x_timestep, syn1, mem1)
                # spk2, syn2, mem2, = self.slstm2(spk1, syn2, mem2)
                # spk3, syn3, mem3, = self.slstm3(spk2, syn3, mem3)
                spkout, synout, memout, = self.slstm_out(spk1, synout, memout)

                spkout_rec.append(spkout)
                memout_rec.append(memout)
            return torch.stack(spkout_rec).permute(1, 0, 2), torch.stack(memout_rec).permute(1, 0, 2)
        else:
            num_steps = x.size()[0]
            for step in range(num_steps):
                x_timestep = x[step]
                spk1, syn1, mem1, = self.slstm1(x_timestep, syn1, mem1)
                spk2, syn2, mem2, = self.slstm2(spk1, syn2, mem2)

                spkout_rec.append(spk2)
                memout_rec.append(mem2)

            return torch.stack(spkout_rec), torch.stack(memout_rec)

class DualInputNet_Spike(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_out) -> None:
        super().__init__()
        self.hidden_size = num_out * 2
        self.num_directions = 1
        self.spike_rnn = SpikeNet(num_inputs, num_hidden, num_out, batch_first=True)
        self.rnn = nn.LSTM(num_out, num_out, num_layers=2, bidirectional=False, dropout=0.2, batch_first=True)
        
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=num_out*2, out_features=num_out),
            nn.LeakyReLU(),
            nn.Linear(in_features=num_out, out_features=1),
        )
        
        # attention params
        self.attention_size = self.hidden_size
        self.w_omega = Variable(
            torch.zeros(self.hidden_size * self.num_directions, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))
        self.sequence_length = 100
        
    # (bs, seq_len, hidden_size)  => (bs, hidden_size) 
    def attention_net(self, lstm_output, device):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size * self.num_directions])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output
    
    def forward(self, x1, x2):
        # x1 = x1.permute(1, 0, 2)
        # x2 = x2.permute(1, 0, 2)
        # x1 = spikegen.delta(x1)
        # x2 = spikegen.delta(x2)
        # x1 = spikegen.rate(x1, time_var_input=True).permute(1, 0, 2)
        # x2 = spikegen.rate(x2, time_var_input=True).permute(1, 0, 2)
        
        x1_spike, x1_umem = self.spike_rnn(x1) # (bs, time_step, num_out)
        x2_spike, x2_umem = self.spike_rnn(x2)
        
        x1, (_, _) = self.rnn(x1_umem)
        x2, (_, _) = self.rnn(x2_umem) 
        concat_input = torch.concat((x1, x2), 2)
        atten_out = self.attention_net(concat_input, concat_input.device) # (bs, num_out)
        out = self.output_layer(atten_out)
        return out

if __name__ == '__main__':
    
    inputs = (torch.randn((1024, 100, 300)) + 3.0).cuda()
    targets = torch.concat((torch.zeros((512,)), torch.ones((512,))*2), dim=0).cuda()

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
    # loss_func = nn.L1Loss(reduction='mean')
    loss_func = nn.MSELoss()
    spiking = DualInputNet_Spike(300, 100, 50).to('cuda')
    optimizer = torch.optim.Adam(spiking.parameters(), lr=1e-4, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.5)
    
    history = []
    n_batch = len(dataloader)
    for epoch in tqdm(range(2000)):
        total_loss = 0    
        for bs_x, bs_y in dataloader:
            out = spiking(bs_x, bs_x).flatten()
            # print(out.size())
            # print(bs_y.size())
            loss_value = loss_func(out, bs_y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            total_loss += float(loss_value)
        history.append(total_loss / n_batch)
        print('loss: ', history[-1])
    # scheduler.step()
    # print(history)
    import matplotlib.pyplot as plt
    
    plt.plot(history)
    plt.savefig('./loss.png')
  