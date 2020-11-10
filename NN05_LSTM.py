import torch
import torch.nn as nn
import files as f
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Model(nn.Module):
    def __init__(self, input_features, hidden_size):
        super(Model, self).__init__()
        lstm_input_size = input_features
        self.lstm01 = nn.LSTM(lstm_input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear01 = nn.Linear(hidden_size, 4)

    def forward(self, x, states):
        h, c = states
        out, (h, c) = self.lstm01(x)
        out = self.linear01(out.view(out.size(0)*out.size(1), out.size(2)))
        return out, states


torch.manual_seed(0)

consideration_length = []
test_length = 400
skip = 50
input_size = 40
future_days = 2
skip_future_days = 0
integral_length = 4

bit11 = "11BIT.csv"
cl_bit = 2200
pko = "PKOBP.csv"
cl_pko = 5300
orlen = "PKNORLEN.csv"
cl_pkn = 4900
alior = "ALIOR.csv"
cl_alior = 1700
ccc = "CCC.csv"
cl_ccc = 3500
polsat = "POLSAT.csv"
cl_polsat = 2200
dino = "DINOPL.csv"
cl_dino = 600
jsw = "JSW.csv"
cl_jsw = 2000
kghm = "KGHM.csv"
cl_kghm = 5500
lpp = "LPP.csv"
cl_lpp = 4400
lotos = "LOTOS.csv"
cl_lotos = 3500
mbk = "MBANK.csv"
cl_mbk = 6400

names = []
names.append(bit11)
consideration_length.append(cl_bit)
# names.append(pko)
# consideration_length.append(cl_pko)
# names.append(orlen)
# consideration_length.append(cl_pkn)
# names.append(alior)
# consideration_length.append(cl_alior)
# names.append(ccc)
# consideration_length.append(cl_ccc)
# names.append(polsat)
# consideration_length.append(cl_polsat)
# names.append(dino)
# consideration_length.append(cl_dino)
# names.append(jsw)
# consideration_length.append(cl_jsw)
# names.append(kghm)
# consideration_length.append(cl_kghm)
# names.append(lpp)
# consideration_length.append(cl_lpp)
# names.append(lotos)
# consideration_length.append(cl_lotos)
# names.append(mbk)
# consideration_length.append(cl_mbk)

for cl, cons_length in enumerate(consideration_length):
    consideration_length[cl] += input_size + integral_length

input_data = []
labels = []
real_prices = []
val_input_data = []
val_labels = []
bases = []

for name, cons_length in zip(names, consideration_length):
    input_data_raw, labels_raw, bases_raw, real_prices_raw = f.open_stock_with_integral(name, cons_length, input_size,
                                                        integral_length, future_days, skip_future_days, skip)
    val_input_data.extend(input_data_raw[-test_length:])
    val_labels.extend(labels_raw[-test_length:])
    input_data.extend(input_data_raw[:-test_length])
    labels.extend(labels_raw[:-test_length])
    bases.extend(bases_raw[-test_length:])
    real_prices.extend(real_prices_raw[-test_length:])

input_data = np.array(input_data)
labels = np.array(labels)
val_input_data = np.array(val_input_data)
val_labels = np.array(val_labels)

input_data = torch.tensor(input_data)
labels = torch.tensor(labels).view(labels.shape[0], -1)
val_input_data = torch.tensor(val_input_data)
val_labels = torch.tensor(val_labels).view(val_labels.shape[0], -1)
bases = torch.tensor(bases)
real_prices = torch.tensor(real_prices)

data_set_training = Dataset(input_data, labels)
data_set_test = Dataset(val_input_data, val_labels)

train_loader = DataLoader(dataset=data_set_training,
                          batch_size=64)
test_loader = DataLoader(dataset=data_set_test,
                         shuffle=False)

input_features = 5 * input_size
output_features = 4 * future_days
hidden_size = 200
net = Model(input_features, hidden_size)

criterion = torch.nn.SmoothL1Loss(size_average=True)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.99)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

epochs = 8

acc_by_time = []
loss_by_time = []
for e in range(epochs):
    net.train()
    outs = []
    for i, l in train_loader:
        if i.size(0) == 64:
            i = i.float()
            l = l.float()
            states = (torch.zeros(1, 64, hidden_size),
                      torch.zeros(1, 64, hidden_size))
            i = i.view(64, 1, -1)
            l = l.view(64, -1)
            outputs, states = net(i, states)
            loss = criterion(outputs, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if (e + 1) % 1 == 0:
        net.eval()
        b = []
        accumulated_loss = []
        for i, l in test_loader:
            if i.size(1) == 64:
                i = i.float()
                l = l.float()
                states = (torch.zeros(1, i.size(1), hidden_size),
                          torch.zeros(1, i.size(1), hidden_size))
                i = i.view(64, 1, -1)
                l = l.view(64, -1)
                b.append(l)
                outputs, states = net(i, states)
                outs.extend(outputs.detach().numpy())
                loss = criterion(outputs, l)
                accumulated_loss.append(loss)

        outs = np.array(outs)
        outs = (outs > 0.0)
        a = outs
        b = np.array(data_set_test.y)
        b = b > 0.0
        #b = np.array(torch.cat(b))
        correctness = torch.tensor(a == b)
        accuracy = correctness.float().mean()
        acc_by_time.append(accuracy)
        accumulated_loss = torch.tensor(accumulated_loss).mean()
        loss_by_time.append(accumulated_loss.item())
        print("Epoch {}/{} Loss: {:.10f} Accuracy: {:.3f}".format(e + 1, epochs, accumulated_loss.item(), accuracy.item()))

    scheduler.step()

print(max(acc_by_time).item())
plt.plot(acc_by_time)
plt.plot(loss_by_time)
plt.show()