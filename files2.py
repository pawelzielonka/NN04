import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import signal
from tsmoothie.smoother import *
import copy


class DataSet:
    def __init__(self, x, y, data_x, data_y, sums_complementary, base_prices):
        self.x = x
        self.y = y
        self.data_x = data_x
        self.data_y = data_y
        self.sums_complementary = sums_complementary
        self.base_prices = base_prices

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def normalize_price(value, in_high, out_high, clip):
    out_low = 0
    in_low = -in_high
    fraction_01 = (out_low - out_high) / (in_low - in_high)
    out = fraction_01 * value + (out_low - fraction_01 * in_low)
    if clip:
        out = np.clip(out, out_low, out_high)
    return np.array(out)
    # fraction_01 = out_high / in_high
    # out = fraction_01 * value
    # if clip:
    #     out = np.clip(out, -1.0, 1.0)
    # return np.array(out)


def normalize_volume(value):
    max_v = np.max(value)
    min_v = np.min(value)
    return (value - min_v) / (max_v - min_v)


def price_relation(value, base_price):
    return (value / base_price - 1) * 100


def price_relation_restore(relative_price, base_price):
    return base_price.reshape(base_price.shape[0], base_price.shape[1], 1) * (1 + relative_price / 100)


def restore_from_ma(price_relative, sums_complementary, ma_length):
    return ma_length * price_relative - sums_complementary


def volume_relation(volume, base_volume, relative_limit):
    relative_volume = volume / base_volume
    for i, v in enumerate(relative_volume):
        for j, vol in enumerate(v):
            if vol > relative_limit:
                relative_volume[i, j] = relative_limit
            elif vol < 1 / relative_limit:
                relative_volume[i, j] = 1 / relative_limit
    return relative_volume


def cast_price(price_list):
    prices_casted = np.zeros((len(price_list)))
    for i, p in enumerate(price_list):
        try:
            prices_casted[i] = float(p.replace(",", "."))
        except:
            prices_casted[i] = price_list[i - 1]
        if prices_casted[i] == 0 or math.isnan(prices_casted[i]):
            prices_casted[i] = prices_casted[i - 1]
    return prices_casted


def cast_volume(volume_list):
    volumes_casted = np.zeros((len(volume_list)))
    for i, v in enumerate(volume_list):
        try:
            volumes_casted[i] = float(v)
        except:
            volumes_casted[i] = 0.1
    return volumes_casted


def calculate_ma(ma_length, trend):
    trend_filtered = np.zeros((trend.shape[0]))
    sums_complementary = np.zeros((trend_filtered.shape[0]))
    for step in range(len(trend)):
        if step >= ma_length - 1:
            trend_filtered[step] = np.sum(trend[step - ma_length + 1:step + 1]) / ma_length
            sums_complementary[step] = np.sum(trend[step - ma_length + 2:step + 1])
        else:
            trend_filtered[step] = trend[step]
            sums_complementary[step] = trend[step] * (ma_length - 1)
    return trend_filtered, sums_complementary


def read_merge_save(name_list, new_name):
    dfs = []
    for name in name_list:
        f = pd.read_csv(name, sep=';')
        dfs.append(f)
    new_df = pd.concat(dfs)
    new_df.to_csv(new_name, sep=';')
    print('Merging successful, new shape: ' + str(new_df.shape))


def save_3d_to_csv(name, data):
    data_frame = pd.DataFrame(data.reshape((data.shape[0], -1)))
    data_frame.to_csv(name, sep=';')
    print('Saved ' + str(name))


def prepare_fx_data(name, period=60, split_perc=10, if_split=True, skip=None):
    if skip == None:
        skip = period
    print('Reading: ' + str(name) + '......')
    f = pd.read_csv(name, sep=';', dtype=np.str)
    prices_size = int((f.shape[0] - period) / skip)
    #int((f.shape[0] - period + 1) / period)
    prices = np.zeros((prices_size, 4))
    open_p = cast_price(f.open_p)
    max_p = cast_price(f.max_p)
    min_p = cast_price(f.min_p)
    close_p = cast_price(f.close_p)
    #period * prices.shape[0] + 1
    for ip, i in enumerate(range(period, skip * prices.shape[0] + period + skip - 1, skip)):
        if ip < prices.shape[0]:
            prices[ip, 0] = open_p[i - period]
            prices[ip, 1] = np.max(max_p[i - period:i])
            prices[ip, 2] = np.min(min_p[i - period:i])
            prices[ip, 3] = close_p[i - 1]

    if if_split:
        cutoff_length = int((f.shape[0] - 60) * split_perc / 100)
        print('First step for dev set: ' + str(f.timest[f.shape[0] - 2 * cutoff_length]))
        print('First step for test set: ' + str(f.timest[f.shape[0] - cutoff_length]))

    return prices


def open_stock(name, steps_total, ma_length, steps, in_high, future_days, split_perc, split, fx=False,
               period=60, skip=None, if_shuffle=True, predictions=False, data=None, filter=False):
    steps_total += steps + future_days
    open_prices, max_prices, min_prices, close_prices, volumes = [], [], [], [], []

    if not fx:
        f = open(name, "r")
        line_list = f.readlines()[1:-1]

        for ln in range(steps_total):
            d = -1 * steps_total + ln
            line = line_list[d].split(",")[4:-1]
            open_prices.append(line[0])
            max_prices.append(line[1])
            min_prices.append(line[2])
            close_prices.append(line[3])
            if len(line) == 5:
                volumes.append(line[4])
            else:
                volumes.append("1")

        open_prices = cast_price(open_prices)
        max_prices = cast_price(max_prices)
        min_prices = cast_price(min_prices)
        close_prices = cast_price(close_prices)
        volumes = cast_volume(volumes)

        data = np.array([open_prices, max_prices, min_prices, close_prices, volumes]).reshape(5, -1)
    else:
        if predictions is not True:
            data = prepare_fx_data(name, period=period, split_perc=split_perc, if_split=split, skip=skip)
        data = np.moveaxis(data, 0, -1)

    _data_filtered = [None] * data.shape[0]
    if not fx:
        _sums_complementary = [None] * (data.shape[0] - 1)
    else:
        _sums_complementary = [None] * data.shape[0]

    for i, column in enumerate(data):
        if not fx:
            _length = len(data) - 1
        else:
            _length = len(data)
        if not filter:
            if i < _length:
                _data_filtered[i], _sums_complementary[i] = calculate_ma(ma_length, column)
            else:
                _data_filtered[i] = volumes
        else:
            _data_filtered = np.zeros(data.shape)
            _sums_complementary = np.zeros(data.shape)
    if not fx:
        # datafiltered niepewny sprawdzic z tym co jest w fx
        _data_filtered = np.array(_data_filtered).reshape((len(_data_filtered), _data_filtered[0].shape[0]))
        _sums_complementary = np.moveaxis(np.array(_sums_complementary), 0, -1)[:-1, :]
        _data = np.moveaxis(data, 0, -1)
        _data_filtered = np.moveaxis(_data_filtered, 0, -1)
    else:
        _data_filtered = np.moveaxis(np.array(_data_filtered), 0, -1)
        _sums_complementary = np.moveaxis(np.array(_sums_complementary), 0, -1)
        _data = np.moveaxis(data, 0, -1)

    if fx:
        feature_count = 4
    else:
        feature_count = 5
    _x = np.zeros((int((_data_filtered.shape[0] - steps - future_days)), future_days, steps * feature_count))
    _y = np.zeros((int((_data_filtered.shape[0] - steps - future_days)), future_days, feature_count))
    data_x = np.zeros(_x.shape)
    data_y = np.zeros(_y.shape)
    sums_complementary = np.zeros((_x.shape[0], future_days, 4))

    for series in range(_x.shape[0]):
        for fd in range(future_days):
            # _x[series, fd, :] = _data_filtered[series + fd:series + fd + steps, :].reshape(-1)
            _x[series, fd, :] = _data_filtered[series:series + steps, :].reshape(-1)
            _y[series, fd, :] = _data_filtered[series + fd + steps, :].reshape(-1)

            # data_x[series, fd, :] = _data[series + fd:series + fd + steps, :].reshape(-1)
            data_x[series, fd, :] = _data[series:series + steps, :].reshape(-1)
            data_y[series, fd, :] = _data[series + fd + steps, :].reshape(-1)
            # sums_complementary[series, fd, :] = _sums_complementary[series + fd + steps - 1, :].reshape(-1)
            sums_complementary[series, fd, :] = _sums_complementary[series + steps - 1, :].reshape(-1)

    x = np.zeros(_x.shape)
    y = np.zeros((x.shape[0], x.shape[1], 4))
    base_prices = np.zeros((x.shape[0], x.shape[1], 1))
    data_std = np.zeros((_x.shape[0], 1))

    for series in range(x.shape[0]):
        if not fx:
            base_price = _x[series, :, -2].reshape((future_days, 1))
            base_prices[series, :, :] = base_price
            relative_limit = 10
            base_volume = _x[series, :, -1].reshape((future_days, 1))

            indices = np.arange(4, steps * feature_count, feature_count)
            relative_prices_x = price_relation(np.delete(_x[series, :, :], indices, axis=1), base_price)
            normalized_prices_x = normalize_price(relative_prices_x, in_high, 1, clip=False)
            relative_volumes_x = volume_relation(_x[series, :, 4::5], base_volume, relative_limit)
            normalized_volumes_x = normalize_volume(relative_volumes_x)
            normalized_volumes_x[1:, :] = 0
            x[series] = np.concatenate((normalized_prices_x, normalized_volumes_x), axis=1)

            relative_prices_y = price_relation(_y[series, :, :-1], base_price)
            normalized_prices_y = normalize_price(relative_prices_y, in_high, 1, clip=True)
            y[series] = normalized_prices_y
            # restored = restore_prices(normalized_prices_y, base_price, sums_complementary[series], in_high, ma_length)
            # series += 0
        else:
            if not filter:
                base_price = _x[series, 0, -1].reshape((-1, 1))
                base_prices[series, :, :] = base_price
                relative_prices_x = price_relation(_x[series, :, :], base_price)
                normalized_prices_x = normalize_price(relative_prices_x, in_high, 1, clip=False)
                x[series] = normalized_prices_x

                relative_prices_y = price_relation(_y[series, :, :], base_price)
                normalized_prices_y = normalize_price(relative_prices_y, in_high, 1, clip=True)
                y[series] = normalized_prices_y
                # restored = restore_prices(normalized_prices_y, np.moveaxis(base_price, 0, -1), sums_complementary[series], in_high, ma_length)
                # series += 0
            else:
                _range = np.zeros((4, steps + future_days))
                filtered_range = np.zeros(_range.shape)

                for i in range(4):
                    _range[i, :-future_days] = data_x[series, 0, i::4]
                    _range[i, -future_days:] = data_y[series, :, i]

                    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                    # ax1.plot(_range[i, :])
                    # ax2.plot(filtered_range[i, :])
                    # plt.show()

                smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother.smooth(_range)
                filtered_range = smoother.smooth_data

                base_price = filtered_range[-1, -future_days - 1].reshape((-1, 1))
                base_prices[series, :, :] = base_price
                relative_prices_x = price_relation(filtered_range[:, :-future_days], base_price)
                normalized_prices_x = normalize_price(relative_prices_x, in_high, 1, clip=False)
                x[series, :, :] = normalized_prices_x.reshape(-1)

                relative_prices_y = price_relation(filtered_range[:, -future_days:], base_price)
                normalized_prices_y = normalize_price(relative_prices_y, in_high, 1, clip=True)
                y[series, :, :] = np.moveaxis(normalized_prices_y, 0, -1)

                # a = filtered_range[-1, :]
                # smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                # smoother.smooth(_range[-1, :-future_days])
                # b = smoother.smooth_data
                #
                # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
                # ax1.plot(a)
                # ax2.plot(b[-1, :])
                # plt.show()

        data_std[series, 0] = np.std(x[series, 0, :])

    mean = np.mean(x)
    std = np.std(x)
    print(name + " " + str(np.round(mean, 4)) + " " + str(np.round(std, 4)))

    if predictions:
        set = DataSet(x[:, :, :], y[:, :, :],
                      data_x[:, :, :], data_y[:, :, :],
                      sums_complementary[:, :, :],
                      base_prices[:, :, :])
        return set
    else:
        if split:
            set_length = x.shape[0]
            cutoff_length = int((set_length - steps) * split_perc / 100)

            if if_shuffle:
                x, y, data_x, data_y, sums_complementary, base_prices = shuffle_unison([x, y, data_x, data_y,
                                                                                        sums_complementary, base_prices],
                                                                                       cutoff_length)

            train_set = DataSet(x[:-2 * cutoff_length, :, :], y[:-2 * cutoff_length, :, :],
                                data_x[:-2 * cutoff_length, :, :], data_y[:-2 * cutoff_length, :, :],
                                sums_complementary[:-2 * cutoff_length, :, :],
                                base_prices[:-2 * cutoff_length, :, :])
            test_set = DataSet(x[-2 * cutoff_length:-cutoff_length, :, :], y[-2 * cutoff_length:-cutoff_length, :, :],
                               data_x[-2 * cutoff_length:-cutoff_length, :, :],
                               data_y[-2 * cutoff_length:-cutoff_length, :, :],
                               sums_complementary[-2 * cutoff_length:-cutoff_length, :, :],
                               base_prices[-2 * cutoff_length:-cutoff_length, :, :])
            dev_set = DataSet(x[-cutoff_length:, :, :], y[-cutoff_length:, :, :],
                              data_x[-cutoff_length:, :, :], data_y[-cutoff_length:, :, :],
                              sums_complementary[-cutoff_length:, :, :],
                              base_prices[-cutoff_length:, :, :])

            return train_set, test_set, dev_set, None
        else:
            set_length = x.shape[0]
            cutoff_length = int((set_length - steps) * split_perc / 100)

            set = DataSet(x[:cutoff_length, :, :], y[:cutoff_length, :, :],
                          data_x[:cutoff_length, :, :], data_y[:cutoff_length, :, :],
                          sums_complementary[:cutoff_length, :, :],
                          base_prices[:cutoff_length, :, :])
            return set, set, set, data_std


def restore_prices(model_output, base_price, sums_complementary, out_high, ma_length):
    price_percentage_change = normalize_price(model_output, 1, out_high=out_high, clip=False)
    price_relative = price_relation_restore(price_percentage_change, base_price)
    price = restore_from_ma(price_relative, sums_complementary, ma_length)
    return price_percentage_change, price_relative, price


def shuffle_unison(arrays_to_shuffle, cutoff):
    if cutoff == 0:
        randomize = np.arange(arrays_to_shuffle[0].shape[0])
        np.random.shuffle(randomize)
    else:
        size = arrays_to_shuffle[0].shape[0]
        aranged_start = size - cutoff
        randomize = np.arange(aranged_start)
        np.random.shuffle(randomize)
        aranged = np.arange(aranged_start, size)
        randomize = np.concatenate([randomize, aranged])
    arrays_shuffled = []
    for array in arrays_to_shuffle:
        arrays_shuffled.append(array[randomize])
    return arrays_shuffled
