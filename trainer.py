import torch
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim as optim
from scipy import fft
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import numpy as np
import os
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, net, batch_size, logdir):
        """
        The classifier used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
        """

        self.best_eer = 1
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net = net
        self.epoch_counter = 0
        self.batch_size = batch_size
        self.epochs = None
        self.c = None
        self.l2 = None
        self.use_cuda = torch.cuda.is_available()
        self.writer = SummaryWriter(logdir)

    def _train_epoch(self, train_loader, optimizer, criterion):
        self.net.train()
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(self.epoch_counter + 1, self.epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for iteration, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.net.forward(inputs)
                optimizer.zero_grad()
                targets = targets.squeeze_()
                loss, w_loss = criterion(logits, targets, self.net.wavelet.hi, self.net.wavelet.lo, self.c, self.l2)
                # loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                _, preds = torch.max(logits, 1)

                targets = targets.to("cpu", torch.double).detach().numpy()
                preds = preds.to("cpu", torch.double).detach().numpy()

                acc = accuracy_score(targets, preds)
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             acc='{0:1.5f}'.format(acc)))
                self.writer.add_scalar('loss', loss.item())
                self.writer.add_scalar("Regularisation", w_loss.item())
                self.writer.add_scalar('acc', acc)
                pbar.update(1)
        return loss.item(), acc, w_loss.item()

    def _validate_epoch(self, val_loader, criterion):
        self.net.eval()
        it_count = len(val_loader)
        all_targets = np.zeros((self.batch_size * (it_count - 1)))
        all_preds = np.zeros((self.batch_size * (it_count - 1)))
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for it, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.net(inputs)
                targets = targets.squeeze_()
                loss, w_loss = criterion(logits, targets, self.net.wavelet.hi, self.net.wavelet.lo, self.c, self.l2)
                _, preds = torch.max(logits, 1)
                targets = targets.to("cpu", torch.double).detach().numpy()
                preds = preds.to("cpu", torch.double).detach().numpy()
                step_acc = accuracy_score(targets, preds)
                if it == it_count - 1:
                    all_targets = np.hstack((all_targets, targets))
                    all_preds = np.hstack((all_preds, preds))
                else:
                    all_targets[it * self.batch_size:it * self.batch_size + self.batch_size] = targets
                    all_preds[it * self.batch_size:it * self.batch_size + self.batch_size] = preds
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             acc='{0:1.5f}'.format(step_acc)))
                pbar.update(1)
                self.writer.add_scalar('val_loss', loss.item())
                self.writer.add_scalar('val_acc', step_acc)
        acc = accuracy_score(all_targets, all_preds)
        print(logits.size())
        print(all_preds)
        print(all_targets)
        # print(all_targets.mean())
        # print((all_targets == all_preds).mean())
        return loss.item(), acc

    def visualize_filters(self):
        hi_f = np.abs(fft(self.net.wavelet.hi[0, 0, :].cpu().data.numpy()))
        lo_f = np.abs(fft(self.net.wavelet.lo[0, 0, :].cpu().data.numpy()))
        n = hi_f.shape[-1]
        plt.plot(np.arange(n // 2 + 1) / (n // 2), lo_f[:n // 2 + 1])
        plt.plot(np.arange(n // 2 + 1) / (n // 2), hi_f[:n // 2 + 1])
        plt.show()

    def _run_epoch(self, train_loader, val_loader,
                   optimizer, criterion, lr_scheduler):

        # Run a train pass on the current epoch
        train_loss, train_acc, w_loss = self._train_epoch(train_loader, optimizer, criterion)

        val_loss, val_acc = self._validate_epoch(val_loader, criterion)

        if val_acc < self.best_eer:
            print("saving model with acc = {:03f}".format(val_acc))
            self.best_eer = val_acc
            torch.save(self.net.state_dict(), "best_model val_acc = {:03f}".format(val_acc))
        lr_scheduler.step(val_acc)
        # Reduce learning rate if needed

        print("train_loss = {:03f}, train_acc = {:03f}, wavelet_loss = {:1.4}\n"
              "val_loss   = {:03f}, val_acc   = {:03f}"
              .format(train_loss, train_acc, w_loss, val_loss, val_acc))
        self.visualize_filters()
        self.epoch_counter += 1

    def train(self, train_loader, val_loader, epochs, c, l2, lr=0.01, weight_decay=0.1, patience=10):
        self.c = c
        self.l2 = l2
        self.net.to(self.device)
        if self.epochs != None:
            self.epochs += epochs
        else:
            self.epochs = epochs
            self.epoch_counter = 0
        criterion = self.net.criterion

        optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.7, verbose=True, threshold=1e-4)
        # lr_scheduler = StepLR(optimizer, patience, gamma=0.5, last_epoch=-1)
        for epoch in range(epochs):
            self._run_epoch(train_loader, val_loader, optimizer, criterion, lr_scheduler)

    def test(self, test_loader, model_name, exp_name):
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(model_name))
        it_count = len(test_loader)
        scores = np.zeros((self.batch_size*it_count-1))
        all_targets = np.zeros((self.batch_size*it_count-1))
        for it, (inputs, targets) in enumerate(test_loader):
            if self.use_cuda:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

            inputs, targets = Variable(inputs), Variable(targets)
            logits = self.net(inputs)
            targets = targets.squeeze_()
            if self.use_cuda:
                targets = targets.to("cpu", torch.double)
                logits = logits.to("cpu", torch.double)
            logits = logits.detach().numpy()
            targets = targets.detach().numpy()
            # print(scores.shape)
            if it == it_count-1:
                scores = np.hstack((scores, logits[:, 1] - logits[:, 0]))
                all_targets = np.hstack((all_targets, targets))
            else:
                scores[it*self.batch_size:it*self.batch_size+self.batch_size] = logits[:, 1] - logits[:, 0]
                all_targets[it*self.batch_size:it*self.batch_size+self.batch_size] = targets
        if not os.path.exists(os.path.join("experiments", exp_name)):
            os.mkdir(os.path.join("experiments", exp_name))
        print(scores.shape, all_targets.shape)
        stats_a = get_eer_stats(scores[all_targets > 0], scores[all_targets == 0])
        print("test_eer = {:03f}".format(stats_a.eer))
        plot_eer_stats([stats_a], ['A'], save_path=os.path.join("experiments", exp_name))
