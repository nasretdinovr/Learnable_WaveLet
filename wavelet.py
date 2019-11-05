import math
import torch
from torch import nn
import torch.nn.functional as F

class Wavelet(nn.Module):
    """Class for computing learnable stationary wavelet transform
            Args:
            n_levels (int): decomposition level of wavelet transform

            filer_size (Tensor): size of first low-pass and high-pass filters

        """
    def __init__(self, num_layers, filer_size,  disp=False):
        super(Wavelet, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.n_levels = num_layers
        self.filer_size = filer_size
        self.filter_bank = []*(self.n_levels+1)
        self.initialize_filters()

    def initialize_filters(self):
        """Initializes filters with xavier uniform distribution and imposes conditions on them in order to be
        wavelet-like
        """

        hi = torch.Tensor(1, 1, self.filer_size)
        nn.init.xavier_uniform_(hi)
        hi = hi - hi.mean()
        self.hi = nn.Parameter(hi / torch.sqrt(self.energy(hi)))
        lo = self.reverse(hi)
        odd = torch.arange(1, lo.size(2) - 1, 2).long()
        lo[:, :, odd] = lo[:, :, odd] * -1
        self.lo = nn.Parameter(lo)

    def energy(self, filter):
        """Computes energy of filter
                Args:
                    filter (Tensor: a filter whose energy to be calculated
        """

        return filter.pow(2).sum()

    def compute_next_level_filer(self, up_filer, lo_filter):
        """Computes (k)th layer low/high-pass filer
        Args:
            up_filer (Tensor): stretched by 2^(k-1) initial low/high-pass filter
            lo_filter (Tensor): (k-1)th layer low-pass filter
        Returns:
            Tensor: stretched by 2^k initial low/high-pass filter
            Tensor: kth layer low/high-pass filter
        """

        pad = lo_filter.size(2) - 1
        upsampeled = self.upsample[:, :, :-1]
        return upsampeled, F.conv1d(F.pad(upsampeled, (pad, pad)), self.reverse(lo_filter))

    def compute_all_filters(self):
        """Computes all high-pass filters for 1-self.n_levels decomposition layers and (self.n_levels)th low-pass
        filter
        """
        self.filter_bank[0] = self.hi
        accumulated_lo = self.lo
        up_lo, up_hi = self.hi, self.lo
        for i in range(1, self.n_levels):
            up_hi, self.filter_bank[i] = self.compute_next_level_filer(up_hi, accumulated_lo)
            last_lo, accumulated_lo = self.compute_next_level_filer(up_lo, accumulated_lo)
        self.filter_bank[self.n_levels] = accumulated_lo

    @staticmethod
    def periodized_extension(x, pad):
        """Inserts [x[x.size() - pad], x[x.size() - pad - 1, ..., x.size()] to the start of the given signal x and
        [x[1], x[2], ... , x[pad]] to the end of x. If signal length is odd last sample is added to end of the signal.

        Args:
            x (Tensor): signal to be extended
            pad (Tensor): length of extension
        """

        if x.size(2) % 2 != 0:
            x = torch.cat((x, x[:, :, -1:]), dim=2)
        return torch.nn.functional.pad(x, (pad, pad), mode='circular')

    @staticmethod
    def reverse(x):
        """ Reverses given signal
        Args:
            x (Tensor): signal to be reversed
        """

        idx = torch.arange(x.size(2) - 1, -1, -1).long()
        return x[:, :, idx]

    def upsample(self, x):
        """Inserts zeros between elements of given signal
        Args:
            x (Tensor): signal to be upsampled
        """

        upsample_filter = torch.zeros(1, 1, 2).to(self.device)
        upsample_filter[0, 0, 0] = 1
        return F.conv_transpose1d(x, upsample_filter, stride=2)

    @staticmethod
    def keep(x, length, start):
        """Extracts vector of given length from a signal beginning with start point
        Args:
            x (Tensor): signal
            length (int): length of output vector
            start (int): starting point output vector is extracted from
        """

        return x[:, :, start:start + length]

    def reconstruction(self, encoding, lo_r, hi_r):
        cD = encoding[:, :-1, :]
        cA = encoding[:, -1:, :]
        _, num_levels, s = cD.size()
        for i in range(num_levels - 1, -1, -1):
            step = int(math.pow(2, i))
            last = step
            for first in range(0, last):
                idx = torch.arange(first, s, step).long()
                lon = idx.size(0)
                sub_idx = idx[torch.arange(0, lon - 1, 2).long()]
                x1 = self.idwt(cA[:, :, sub_idx], cD[:, i:i + 1, sub_idx], lo_r, hi_r, lon, 0)
                sub_idx = idx[torch.arange(1, lon, 2).long()]
                x2 = self.idwt(cA[:, :, sub_idx], cD[:, i:i + 1, sub_idx], lo_r, hi_r, lon, -1)
                cA[:, :, idx] = 0.5 * (x1 + x2)
        return cA

    def upconv(self, x, filter, length):
        lf = filter.size(2)
        y = self.upsample(x)
        y = self.periodized_extension(y, lf // 2)
        y = F.conv1d(F.pad(y, (lf - 1, lf - 1)), self.reverse(filter))
        y = self.keep(y, length, lf - 1)
        return y

    def idwt(self, a, d, lo_r, hi_r, lon, shift):
        y = self.upconv(a, lo_r, lon) + self.upconv(d, hi_r, lon);
        if shift == -1:
            y = torch.cat((y[:, :, -1:], y[:, :, :-1]), dim=2)
        return y

    def forward(self, signal):
        s = signal.size(2)
        cD = torch.Tensor(signal.size(0), self.n_levels, s).zero_().to(self.device)
        cA = torch.Tensor(signal.size(0), self.n_levels, s).zero_().to(self.device)
        lo = self.lo
        hi = self.hi
        for i in range(self.n_levels):
            lf = lo.size(2)
            signal = self.periodized_extension(signal, lf // 2)
            pad = lf - 1
            cD[:, i, :] = self.keep(F.conv1d(F.pad(signal, (pad, pad)), self.reverse(hi)), s, lf).squeeze(1)
            cA[:, i, :] = self.keep(F.conv1d(F.pad(signal, (pad, pad)), self.reverse(lo)), s, lf).squeeze(1)
            lo = self.upsample(lo)
            hi = self.upsample(hi)
            signal = cA[:, i, :].unsqueeze(1)
        return torch.cat((cD, cA[:, -1, :].unsqueeze(1)), dim=1)

