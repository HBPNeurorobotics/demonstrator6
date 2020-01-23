import torch
import torch.nn as nn
from torch.nn import functional as F
from convlstmcell import ConvLSTMCell
from torch.autograd import Variable

class PredNet(nn.Module):
    
    def __init__(self, R_channels, A_channels, device='cpu', t_extrap=float('inf'), scale=4):
        
        super(PredNet, self).__init__()
        self.r_channels = R_channels + (0,)  # for convenience
        self.a_channels = A_channels
        self.n_layers   = len(R_channels)
        self.device     = device
        self.t_extrap   = t_extrap

        for i in range(self.n_layers):

            cell = ConvLSTMCell(2*self.a_channels[i] + self.r_channels[i+1], self.r_channels[i], (3, 3))
            setattr(self, 'cell{}'.format(i), cell)

        for i in range(self.n_layers):

            conv = nn.Sequential(nn.Conv2d(self.r_channels[i], self.a_channels[i], 3, padding=1), nn.ReLU())
            if i == 0: 
                conv.add_module('satlu', SatLU())
            setattr(self, 'conv{}'.format(i), conv)

        self.scale    = scale
        self.upsample = nn.Upsample(scale_factor=scale)
        self.maxpool  = nn.MaxPool2d(kernel_size=scale, stride=scale)
        for l in range(self.n_layers - 1):

            update_A = nn.Sequential(nn.Conv2d(2*self.a_channels[l], self.a_channels[l+1], (3, 3), padding=1), self.maxpool)
            setattr(self, 'update_A{}'.format(l), update_A)

        self.reset_parameters()

    def reset_parameters(self):
        
        for l in range(self.n_layers):
            cell = getattr(self, 'cell{}'.format(l))
            cell.reset_parameters()

    def forward(self, x, nt):

        R_seq = [None]*self.n_layers
        H_seq = [None]*self.n_layers
        E_seq = [None]*self.n_layers

        w, h       = x.size(-2), x.size(-1)
        batch_size = x.size(0)
        for l in range(self.n_layers):

            E_seq[l] = Variable(torch.zeros(batch_size, 2*self.a_channels[l], w, h, device=self.device))
            R_seq[l] = Variable(torch.zeros(batch_size,   self.r_channels[l], w, h, device=self.device))
            w = w//self.scale
            h = h//self.scale

        frame_preds = []
        for t in range(nt):

            # Downward pass
            for l in reversed(range(self.n_layers)):
        
                cell = getattr(self, 'cell{}'.format(l))
                if t == 0:
                    E  = E_seq[l]
                    R  = R_seq[l]
                    hx = (R, R)
                else:
                    E  = E_seq[l]
                    R  = R_seq[l]
                    hx = H_seq[l]
        
                if l == self.n_layers - 1:
                    R, hx = cell(E, hx)
                else:
                    tmp   = torch.cat((E, self.upsample(R_seq[l+1])), 1)
                    R, hx = cell(tmp, hx)
                
                R_seq[l] = R
                H_seq[l] = hx

                if l == self.n_layers-1 and t == self.t_extrap:
                    latent_state = R.detach()

            # Input to the lowest layer
            if t < self.t_extrap:
                A = x[:,t]

            # Upward pass
            for l in range(self.n_layers):
                
                conv  = getattr(self, 'conv{}'.format(l))
                A_hat = conv(R_seq[l])
                if l == 0:
                    frame_pred = A_hat

                if t < self.t_extrap:
                    pos = F.relu(A_hat - A)
                    neg = F.relu(A - A_hat)
                    E   = torch.cat([pos, neg],1)
                    E_seq[l] = E
                    if l < self.n_layers - 1:
                        update_A = getattr(self, 'update_A{}'.format(l))
                        A        = update_A(E)

            frame_preds.append(frame_pred)

        return frame_preds, latent_state


class SatLU(nn.Module):

    def __init__(self, lower=0, upper=255, inplace=False):

        super(SatLU, self).__init__()
        self.lower   = lower
        self.upper   = upper
        self.inplace = inplace

    def forward(self, x):

        return F.hardtanh(x, self.lower, self.upper, self.inplace)


    def __repr__(self):

        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + 'min_val=' + str(self.lower)\
        + ', max_val=' + str(self.upper) + inplace_str + ')'