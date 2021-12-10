import torch
from torch import nn

from wavelet import Wavelet


class Net(nn.Module):
    def __init__(self, wavelet_num_layers, wavelet_kernel_size, n_classes):
        super(Net, self).__init__()
        self.wavelet_num_layers = wavelet_num_layers
        self.wavelet_kernel_size = wavelet_kernel_size
        self.n_classes = n_classes

        self.wavelet = Wavelet(wavelet_num_layers, wavelet_kernel_size)
        self.conv1 = nn.Conv1d(wavelet_num_layers + 1, 10, 3)
        self.pool = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(10, 15, 3)
        self.lastpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(15, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, self.n_classes)

    def criterion(self, outputs, labels, w_hi, w_lo, C, lambda_reg):
        criterion = nn.CrossEntropyLoss()
        kernel_size = w_hi.size(2)

        L1 = (w_hi.pow(2).sum() - 1).pow(2)
        L2 = (w_lo.pow(2).sum() - 1).pow(2)
        en = L1 + L2

        for m in range(1, w_hi.size(2) // 2):
            if w_hi.is_cuda:
                tmp = torch.zeros((1, 1, 1)).cuda()
            else:
                tmp = torch.zeros((1, 1, 1))
            prods = [w_hi[:, :, i] * w_hi[:, :, i + 2 * m] for i in range(kernel_size - 2 * m)]
            for n in prods: tmp += n
            L1 += tmp[0, 0, 0].pow(2)

        for m in range(1, w_hi.size(2) // 2):
            if w_hi.is_cuda:
                tmp = torch.zeros((1, 1, 1)).cuda()
            else:
                tmp = torch.zeros((1, 1, 1))
            prods = [w_lo[:, :, i] * w_lo[:, :, i + 2 * m] for i in range(kernel_size - 2 * m)]
            for n in prods: tmp += n
            L2 += tmp[0, 0, 0].pow(2)

        L3 = w_hi.sum().pow(2)

        L4 = (w_lo.sum() - 2 ** (1 / 2)).pow(2)

        if w_hi.is_cuda:
            L5 = torch.zeros((1)).cuda()
        else:
            L5 = torch.zeros((1))
        for m in range(w_hi.size(2) // 2):
            if w_hi.is_cuda:
                tmp = torch.zeros((1, 1, 1)).cuda()
            else:
                tmp = torch.zeros((1, 1, 1))
            prods = [w_lo[:, :, i] * w_hi[:, :, i + 2 * m] for i in range(kernel_size - 2 * m)]
            for n in prods: tmp += n
            L5 += tmp[0, 0, 0].pow(2)

        if w_hi.is_cuda:
            l2_reg = torch.tensor(0.).cuda()
        else:
            l2_reg = torch.tensor(0.)
        for i, par in enumerate(self.parameters()):
            if i < 2:
                continue
            l2_reg += torch.norm(par)
        CE_Loss = criterion(outputs, labels)
        return CE_Loss + C * (L1 + L2 + L3 + L4 + L5) + lambda_reg * l2_reg, (L1 + L2 + L3 + L4 + L5)

    def forward(self, x):
        encoded_x = self.wavelet(x)
        print(torch.mean(encoded_x, dim=2))
        output = self.pool(self.conv1(encoded_x))
        output = self.conv2(output)
        output = self.lastpool(output)
        output = self.relu(self.fc1(output.squeeze()))
        return self.fc2(output)