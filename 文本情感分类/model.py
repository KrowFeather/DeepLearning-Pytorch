import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import get_dataloader
from lib import ws, max_len
import torch.nn.functional as F
from torch.optim import Adam


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(len(ws), 100)
        self.fc = nn.Linear(max_len * 100, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(-1, max_len * 100)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)


model = Net()
optimizer = Adam(model.parameters(), lr=0.001)


def train(epoch):
    hist = []
    for idx, (input, target) in enumerate(get_dataloader()):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        hist.append(loss.item())
        optimizer.step()
        print('loss: {}'.format(loss.item()))
    fig = plt.figure(figsize=(10, 6))
    plt.plot(hist)
    fig.show()


if __name__ == '__main__':
    for i in range(1):
        train(epoch=i)
