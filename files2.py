import numpy as np
import math


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
    fraction_01 = out_high / in_high
    out = fraction_01 * value
    if clip:
        out = np.clip(out, -1.0, 1.0)
    return np.array(out)


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
    # for step in range(len(trend) - 1):
    #     if step < len(trend) - ma_length:
    #         trend_filtered[step] = np.sum(trend[step:step + ma_length]) / ma_length
    #         sums_complementary[step] = np.sum(trend[step + 1:step + ma_length])
    #         step += 0
    #     else:
    #         trend_filtered[step] = trend[step]
    #         sums_complementary[step] = trend[step] * (ma_length - 1)
    for step in range(len(trend)):
        if step >= ma_length - 1:
            trend_filtered[step] = np.sum(trend[step - ma_length + 1:step + 1]) / ma_length
            sums_complementary[step] = np.sum(trend[step - ma_length + 2:step + 1])
        else:
            trend_filtered[step] = trend[step]
            sums_complementary[step] = trend[step] * (ma_length - 1)
    return trend_filtered, sums_complementary


def open_stock(name, steps_total, ma_length, steps, in_high, future_days, split_perc, split):
    steps_total += steps + future_days
    open_prices, max_prices, min_prices, close_prices, volumes = [], [], [], [], []

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
    _data_filtered = [None] * data.shape[0]
    _sums_complementary = [None] * (data.shape[0] - 1)

    for i, column in enumerate(data):
        if i < len(data) - 1:
            _data_filtered[i], _sums_complementary[i] = calculate_ma(ma_length, column)
        else:
            _data_filtered[i] = volumes
    _data_filtered = np.array(_data_filtered).reshape((len(_data_filtered), _data_filtered[0].shape[0]))
    _sums_complementary = np.moveaxis(np.array(_sums_complementary), 0, -1)[:-1, :]

    _data = np.moveaxis(data, 0, -1)
    _data_filtered = np.moveaxis(_data_filtered, 0, -1)

    _x = np.zeros((int((_data_filtered.shape[0] - steps - future_days)), future_days, steps * 5))
    _y = np.zeros((int((_data_filtered.shape[0] - steps - future_days)), future_days, 5))
    data_x = np.zeros(_x.shape)
    data_y = np.zeros(_y.shape)
    sums_complementary = np.zeros((_x.shape[0], future_days, 4))

    for series in range(_x.shape[0]):
        for fd in range(future_days):
            _x[series, fd, :] = _data_filtered[series + fd:series + fd + steps, :].reshape(-1)
            _y[series, fd, :] = _data_filtered[series + fd + steps, :].reshape(-1)

            data_x[series, fd, :] = _data[series + fd:series + fd + steps, :].reshape(-1)
            data_y[series, fd, :] = _data[series + fd + steps, :].reshape(-1)
            sums_complementary[series, fd, :] = _sums_complementary[series + fd + steps - 1, :].reshape(-1)
            fd += 0

    x = np.zeros(_x.shape)
    y = np.zeros((x.shape[0], x.shape[1], 4))
    base_prices = np.zeros((x.shape[0], x.shape[1], 1))

    for series in range(x.shape[0]):
        base_price = _x[series, :, -2].reshape((future_days, 1))
        base_prices[series, :, :] = base_price
        relative_limit = 10
        base_volume = _x[series, :, -1].reshape((future_days, 1))

        indices = np.arange(4, steps * 5, 5)
        relative_prices_x = price_relation(np.delete(_x[series, :, :], indices, axis=1), base_price)
        relative_volumes_x = volume_relation(_x[series, :, 4::5], base_volume, relative_limit)
        normalized_prices_x = normalize_price(relative_prices_x, in_high, 1, clip=False)
        normalized_volumes_x = normalize_volume(relative_volumes_x)
        normalized_volumes_x[1:, :] = 0
        x[series] = np.concatenate((normalized_prices_x, normalized_volumes_x), axis=1)

        relative_prices_y = price_relation(_y[series, :, :-1], base_price)
        normalized_prices_y = normalize_price(relative_prices_y, in_high, 1, clip=True)
        y[series] = normalized_prices_y

    if split:
        train_length = int((steps_total - steps) * split_perc / 100)

        train_set = DataSet(x[:train_length, :, :], y[:train_length, :, :],
                            data_x[:train_length, :, :], data_y[:train_length, :, :],
                            sums_complementary[:train_length, :, :],
                            base_prices[:train_length, :, :])
        test_set = DataSet(x[train_length:, :, :], y[train_length:, :, :],
                           data_x[train_length:, :, :], data_y[train_length:, :, :],
                           sums_complementary[train_length:, :, :],
                           base_prices[train_length:, :, :])

        return train_set, test_set
    else:
        set = DataSet(x, y, data_x, data_y, sums_complementary, base_prices)
        return set, _


def restore_prices(model_output, base_price, sums_complementary, out_high, ma_length):
    price_percentage_change = normalize_price(model_output, 1, out_high=out_high, clip=False)
    price_relative = price_relation_restore(price_percentage_change, base_price)
    price = restore_from_ma(price_relative, sums_complementary, ma_length)
    return price_percentage_change, price_relative, price