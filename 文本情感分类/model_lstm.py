import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import lib
from dataset import get_dataloader
from lib import ws


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        self.lstm = nn.LSTM(100, hidden_size=lib.hidden_size, num_layers=lib.num_layers, batch_first=True,
                            bidirectional=lib.bidirectional, dropout=lib.dropout)
        self.fc = nn.Linear(lib.hidden_size * 2, 2)

    def forward(self, x):
        x = self.embedding(x)
        out, (h_n, c_n) = self.lstm(x)
        out_fw = h_n[-2, :, :]
        out_bw = h_n[-1, :, :]
        out = torch.concat([out_fw, out_bw], dim=-1)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)


model = Net()
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists('./model/model.pkl'):
    model.load_state_dict(torch.load('./model/model.pkl'))
    optimizer.load_state_dict(torch.load('./model/optimizer.pkl'))


def train(epoch):
    hist = []
    for idx, (input, target) in tqdm(enumerate(get_dataloader())):
        input = input.to(lib.device)
        target = target.to(lib.device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        hist.append(loss.item())
        optimizer.step()
        print('epoch:{} loss: {}'.format(epoch, loss.item()))
        if idx % 5 == 0:
            torch.save(model.state_dict(), './model/model.pkl')
            torch.save(optimizer.state_dict(), './model/optimizer.pkl')
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist)
    fig.show()


def eval(epoch):
    loss_list = []
    acc_list = []
    for idx, (input, target) in tqdm(enumerate(get_dataloader()), total=len(get_dataloader())):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            loss = F.nll_loss(output, target)
            loss_list.append(loss.item())
            pred = output.max(dim=-1)[-1]
            pred = pred.eq(target).float().mean()
            acc_list.append(pred.cpu().item())
    print('total acc,loss: ', np.mean(acc_list), np.mean(loss_list))


if __name__ == '__main__':
    for i in range(1):
        # train(epoch=i)
        eval(epoch=i)
