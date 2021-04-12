import files2
import FX_model as lstm_model
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime


class Params:
    def __init__(self):
        self.future_days = 3
        self.time_steps = 40
        self.lstm_input_features = [4 * self.time_steps, 80]
        self.lstm_hidden_sizes = [80, 15]
        self.lstm_layers_num = [5, 1]
        self.lstm_bidirectional = [True, False]
        self.linear_hidden_sizes = [self.lstm_hidden_sizes[-1], 300, 100]
        self.scale_high = 0.2
        self.ma_length = 10
        self.period = 120
        self.skip = 120

        # FOR FX
        self.lstm_input_features = [4 * self.time_steps, 200, 150]
        self.lstm_hidden_sizes = [200, 150, 30]
        self.lstm_layers_num = [5, 1, 1]
        self.lstm_bidirectional = [True, False, False]
        self.linear_hidden_sizes = [self.lstm_hidden_sizes[-1], 300, 100]


def accuracy_check(predictions, labels, treshold=0.5):
    predicted_positive = predictions > treshold
    real_positive = labels > treshold

    accuracy, precision, recall, f1_score = get_metrics(predicted_positive, real_positive)

    message = str("Accuracy: " + str(round(accuracy.item(), 2)) + " Precision: " + str(round(precision.item(), 2)) +
                  " Recall: " + str(round(recall.item(), 2)) + " F1 score: " + str(round(f1_score.item(), 2)))
    return message


def get_metrics(predicted_positive, real_positive):
    predicted_negative = ~predicted_positive
    real_negative = ~real_positive

    true_positive = (real_positive * predicted_positive).clone().detach().float().sum()
    false_positive = (real_negative * predicted_positive).clone().detach().float().sum()
    false_negative = (real_positive * predicted_negative).clone().detach().float().sum()
    accuracy = (predicted_positive == real_positive).clone().detach().float().mean()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score


def load_data(params):
    bit11 = "11b.txt"
    cl_bit = 2200
    pko = "pko.txt"
    cl_pko = 3800
    orlen = "pkn.txt"
    cl_pkn = 4900
    alior = "ali.txt"
    cl_alior = 1400
    ccc = "ccc.txt"
    cl_ccc = 3500
    polsat = "cps.txt"
    cl_polsat = 2200
    dino = "dnp.txt"
    cl_dino = 600
    jsw = "jsw.txt"
    cl_jsw = 2000
    kghm = "kgh.txt"
    cl_kghm = 5500
    lpp = "lpp.txt"
    cl_lpp = 4400
    lotos = "lts.txt"
    cl_lotos = 3500
    mbk = "mbk.txt"
    cl_mbk = 6400
    names = []
    steps_nums = []
    names.append(bit11)
    steps_nums.append(cl_bit)
    names.append(pko)
    steps_nums.append(cl_pko)
    names.append(orlen)
    steps_nums.append(cl_pkn)
    names.append(alior)
    steps_nums.append(cl_alior)
    names.append(ccc)
    steps_nums.append(cl_ccc)
    names.append(polsat)
    steps_nums.append(cl_polsat)
    names.append(dino)
    steps_nums.append(cl_dino)
    names.append(jsw)
    steps_nums.append(cl_jsw)
    names.append(kghm)
    steps_nums.append(cl_kghm)
    names.append(lpp)
    steps_nums.append(cl_lpp)
    names.append(lotos)
    steps_nums.append(cl_lotos)
    names.append(mbk)
    steps_nums.append(cl_mbk)

    data_train_list, data_test_list = [], []

    for name, steps_num in zip(names, steps_nums):
        data_train, data_test = files2.open_stock(name=name,
                                                  steps_total=steps_num,
                                                  ma_length=6,
                                                  steps=params.time_steps,
                                                  in_high=5,
                                                  future_days=params.future_days,
                                                  split_perc=90,
                                                  split=True)
        data_train_list.append(data_train)
        data_test_list.append(data_test)

    output_data = []

    for data_list in [data_train_list, data_test_list]:
        x = np.concatenate([data.x for data in data_list])
        y = np.concatenate([data.y for data in data_list])
        d_x = np.concatenate([data.data_x for data in data_list])
        d_y = np.concatenate([data.data_y for data in data_list])
        sums = np.concatenate([data.sums_complementary for data in data_list])
        bases = np.concatenate([data.base_prices for data in data_list])

        output_data.append(files2.DataSet(x, y, d_x, d_y, sums, bases))

    output_data, (mean, std) = normalize_data(output_data)

    return output_data, (mean, std)


def normalize_data(output_data, fx=False, mean=0, std=0):
    if mean == 0 and std == 0:
        print("Mean, std calculated")
    else:
        print("Mean, std copied")
    if not fx:
        if mean == 0 and std == 0:
            mean = [np.mean(output_data[0].x[:, :, :-1]), np.mean(output_data[0].x[:, :, -1])]
            std = [np.std(output_data[0].x[:, :, :-1]), np.std(output_data[0].x[:, :, -1])]
        for i in range(len(output_data)):
            output_data[i].x[:, :, :-1] = (output_data[i].x[:, :, :-1] - mean[0]) / std[0]
            output_data[i].x[:, :, -1] = (output_data[i].x[:, :, -1] - mean[1]) / std[1]
    else:
        if mean == 0 and std == 0:
            mean = np.mean(output_data[0].x[:, :, :])
            std = np.std(output_data[0].x[:, :, :])
        for i in range(len(output_data)):
            output_data[i].x = (output_data[i].x - mean) / std
    print("Mean: " + str(np.round(mean, 4)))
    print("Std: " + str(np.round(std, 4)))
    return output_data, (mean, std)


def make_lists_fx():
    names, steps_nums = [], []
    names.append('USDPLN.csv')
    # names.append('EURUSD.csv')
    return names, steps_nums


def make_single_list_fx():
    names, steps_num = [], []
    names.append('DAT_XLSX_USDPLN_M1_2020.csv')
    # names.append('DAT_XLSX_EURUSD_M1_2020.csv')
    return names, steps_num


def load_data_fx(params, if_multiple=True, if_split=True, perc=10, mean=0, std=0):
    if if_multiple:
        names, steps_nums = make_lists_fx()
    else:
        names, steps_nums = make_single_list_fx()

    data_train_list, data_test_list, data_dev_list = [], [], []

    for name in names:
        data_train, data_test, data_dev, data_std = files2.open_stock(name=name,
                                                                      steps_total=0,
                                                                      ma_length=params.ma_length,
                                                                      steps=params.time_steps,
                                                                      in_high=params.scale_high,
                                                                      future_days=params.future_days,
                                                                      split_perc=perc,
                                                                      split=if_split,
                                                                      fx=True,
                                                                      period=params.period,
                                                                      skip=params.skip,
                                                                      filter=True)
        data_train_list.append(data_train)
        data_test_list.append(data_test)
        data_dev_list.append(data_dev)

    output_data = []

    for data_list in [data_train_list, data_test_list, data_dev_list]:
        x = np.concatenate([data.x for data in data_list])
        y = np.concatenate([data.y for data in data_list])
        d_x = np.concatenate([data.data_x for data in data_list])
        d_y = np.concatenate([data.data_y for data in data_list])
        sums = np.concatenate([data.sums_complementary for data in data_list])
        bases = np.concatenate([data.base_prices for data in data_list])

        output_data.append(files2.DataSet(x, y, d_x, d_y, sums, bases))

    output_data, (mean, std) = normalize_data(output_data, fx=True, mean=mean, std=std)

    return output_data, (mean, std), data_std


def test_data_set(model, criterion, data_set_loader, params, _loss, message):
    for x, y in data_set_loader:
        states = model.return_states(params.lstm_layers_num, params.lstm_bidirectional, x.shape[0], params.lstm_hidden_sizes)
        x = x.type(dtype=torch.float32)
        y = y.type(dtype=torch.float32)
        out, states = model(x, states)
        loss = criterion(out, y)
        _loss.append(loss.item())
        print(message + accuracy_check(out, y))


def train(name, epochs=4, batch_size=64, load_model=False, learning_rate=0.001, name_to_save=''):

    params = Params()

    model = lstm_model.Model(lstm_input=params.lstm_input_features,
                             lstm_hidden=params.lstm_hidden_sizes,
                             lstm_layers_num=params.lstm_layers_num,
                             lstm_bidirectional=params.lstm_bidirectional,
                             linear_hidden_size=params.linear_hidden_sizes,
                             timesteps=params.time_steps,
                             future_days=params.future_days)
    if load_model:
        model = torch.load(name + ".pt")

    print("Trainable parameters of the model: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    #data_train, data_test = load_data(params)
    (data_train, data_dev, data_test), (mean, std), _ = load_data_fx(params, perc=5)
    #(data_test_1, _, _), (mean, std) = load_data_fx(params, if_multiple=False, mean=mean, std=std, perc=30)
    model.add_mean_std(mean, std)
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=data_dev, batch_size=data_dev.__len__())
    test_loader = DataLoader(dataset=data_test, batch_size=data_test.__len__())

    print("Train set length: " + str(data_train.__len__()))
    print("Dev set length: " + str(data_dev.__len__()))
    print("Test set length: " + str(data_test.__len__()))

    criterion = torch.nn.SmoothL1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2, gamma=0.1)

    for e in range(epochs):
        print("\nEpoch: " + str(e))
        model.train()
        train_loss = []
        dev_loss = []
        test_loss = []
        for x, y in train_loader:
            states = model.return_states(params.lstm_layers_num, params.lstm_bidirectional, x.shape[0], params.lstm_hidden_sizes)
            x = x.type(dtype=torch.float32)
            y = y.type(dtype=torch.float32)
            out, states = model(x, states)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if len(train_loss) % 1000 == 0:
                percentage = np.round(100 * len(train_loss) / len(train_loader), 2)
                print("% of epoch: " + str(percentage) + ", loss: " + str(np.round(np.mean(train_loss), 4)))
                print("Batch metrics: " + accuracy_check(out, y))

        model.eval()
        test_data_set(model=model, criterion=criterion, data_set_loader=dev_loader, params=params, _loss=dev_loss,
                      message="Dev metrics: ")
        test_data_set(model=model, criterion=criterion, data_set_loader=test_loader, params=params, _loss=test_loss,
                      message="Test metrics: ")

        print("Train loss: " + str(round(np.mean(train_loss), 4))
              + ", Dev loss: " + str(round(np.mean(dev_loss), 4))
              + ", Test loss: " + str(round(np.mean(test_loss), 4)))
        scheduler.step()

    print()
    name_to_save = name_to_save + ".pt" # + datetime.datetime.now().strftime("%d_%m_%Y_%H_%M") + ".pt"
    # y = y.detach().numpy()
    # out = out.detach().numpy()
    # prices_predicted = files2.restore_prices(out, data_test.base_prices, data_test.sums_complementary, params.scale_high, params.ma_length)
    # real_prices_restored = files2.restore_prices(y, data_test.base_prices, data_test.sums_complementary, params.scale_high, params.ma_length)
    # actual_prices = data_test.data_y[:, :, :]
    torch.save(model, name_to_save)


#train(epochs=10)
