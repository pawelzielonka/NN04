import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, lstm_input, lstm_hidden, lstm_layers_num, lstm_bidirectional, linear_hidden_size, timesteps, future_days):
        super(Model, self).__init__()
        self.lstm = nn.ModuleList()
        self.linear = nn.ModuleList()

        for i, direction in enumerate(lstm_bidirectional):
            if direction:
                if i == len(lstm_bidirectional) - 1:
                    linear_hidden_size[0] *= 2
                lstm_input[i + 1] *= 2

        for lstm_input, hidden, num, bidir in zip(lstm_input, lstm_hidden, lstm_layers_num, lstm_bidirectional):
            self.create_lstm(lstm_input, hidden, num, bidir)

        for l in range(len(linear_hidden_size) - 1):
            self.create_linear(linear_hidden_size[l], linear_hidden_size[l + 1])
        self.create_linear(linear_hidden_size[-1], 4)

        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)
        self.lowdrop = nn.Dropout(p=0.1)

    def forward(self, x, states):
        states_out = []
        out = x
        for lstm, sts in zip(self.lstm, states):
            out, states = lstm(out, sts)
            states_out.append(states)
            out = self.lowdrop(out)
        for i, linear in enumerate(self.linear):
            out = linear(out)
            if i != len(self.linear) - 1:
                out = self.relu(out)
            else:
                out = self.tanh(out)
            if i == 0:
                out = self.dropout(out)
        return out, states_out

    def create_lstm(self, input_features, hidden_size, layers_num, bidirectional):
        lstm = nn.LSTM(input_features, hidden_size, num_layers=layers_num, batch_first=True, bidirectional=bidirectional)
        self.lstm.append(lstm)

    def create_linear(self, input_features, output_features):
        linear = nn.Linear(input_features, output_features)
        nn.init.xavier_uniform_(linear.weight)
        self.linear.append(linear)

    def return_states(self, lstm_layers_num, lstm_bidirectional, sequence_length, lstm_hidden_sizes):
        return [(
                torch.zeros(lstm_layers_num[i] * int(lstm_bidirectional[i].__float__() + 1), sequence_length, lstm_hidden_sizes[i]),
                torch.zeros(lstm_layers_num[i] * int(lstm_bidirectional[i].__float__() + 1), sequence_length, lstm_hidden_sizes[i]))
                for i in range(len(lstm_hidden_sizes))]