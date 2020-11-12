import numpy as np


def open11bit(name, days, input_size):
    open_prices = []
    min_prices = []
    max_prices = []
    close_prices = []
    volumes = []

    f = open(name, "r")
    linelist = f.readlines()
    for ln in range(days):
        d = -1 * days + ln
        line = linelist[d].split(";")
        open_prices.append(line[1])
        max_prices.append(line[2])
        min_prices.append(line[3])
        close_prices.append(line[4])
        if len(line) == 6:
            volumes.append(line[5][:-2])
        else:
            volumes.append("1")
    data = []
    labels = []
    consideration_length = input_size
    for i in range(consideration_length, len(open_prices) - 1):
        line = []
        line += open_prices[i - consideration_length:i]
        line += max_prices[i - consideration_length:i]
        line += min_prices[i - consideration_length:i]
        line += close_prices[i - consideration_length:i]
        line += volumes[i - consideration_length:i]
        data.append(line)
        labels.append(close_prices[i])
    for d in range(len(data)):
        labels[d] = float(labels[d].replace(",", "."))
        for n in range(len(data[d])):
            data[d][n] = float(data[d][n].replace(",", "."))
    for days in range(len(data)):
        if labels[days] > data[days][-input_size - 1]:
            labels[days] = [1, -1]
        else:
            labels[days] = [-1, 1]
        base_price = data[days][-input_size - 1]
        base_volume = data[days][-1]
        for p in range(len(data[days]) - input_size):
            data[days][p] = (data[days][p] / base_price) * 10 - 10
            if data[days][p] > 1:
                data[days][p] = 1
            if data[days][p] < -1:
                data[days][p] = -1
        for v in range(len(data[days]) - input_size, len(data[days])):
            relative_volume = data[days][v] / base_volume
            if relative_volume < 0.1:
                relative_volume = 0.1
            if relative_volume > 2:
                relative_volume = 2
            data[days][v] = relative_volume - 1
        #labels[days] /= base_price


    #data = np.array(data)
    #labels = np.array(labels)

    return  data, labels


def open_stock_with_mean(name, days, input_size, mov_avg_length, future_days, skip_future_days, skip):
    open_prices = []
    min_prices = []
    max_prices = []
    close_prices = []
    volumes = []
    bases = []
    mov_avg_length = int(mov_avg_length)

    f = open(name, "r")
    linelist = f.readlines()
    for ln in range(days):
        d = -1 * days + ln - skip
        line = linelist[d].split(";")
        open_prices.append(line[1])
        max_prices.append(line[2])
        min_prices.append(line[3])
        close_prices.append(line[4])
        if len(line) == 6:
            volumes.append(line[5][:-2])
        else:
            volumes.append("1")

    for d in range(len(close_prices)):
        try:
            open_prices[d] = float(open_prices[d].replace(",", "."))
        except Exception as e:
            open_prices[d] = open_prices[d - 1]
        try:
            max_prices[d] = float(max_prices[d].replace(",", "."))
        except Exception as e:
            max_prices[d] = max_prices[d - 1]
        try:
            min_prices[d] = float(min_prices[d].replace(",", "."))
        except Exception as e:
            min_prices[d] = min_prices[d - 1]
        try:
            close_prices[d] = float(close_prices[d].replace(",", "."))
        except Exception as e:
            close_prices[d] = close_prices[d - 1]
    for d in range(len(volumes)):
        try:
            volumes[d] = float(volumes[d].replace(",", "."))
        except Exception as e:
            volumes[d] = 1

    # moving average calculation
    moving_average_for_day = []
    for i in range(0, len(close_prices)):
        ma = 0
        if i >= mov_avg_length - 1:
            for day in range(0, mov_avg_length):
                ma += float(close_prices[i - day]) / mov_avg_length
        moving_average_for_day.append(ma)

    data = []
    labels = []
    real_prices = []
    consideration_length = input_size
    prev = [0, 0, 0, 0]
    prev[0] = open_prices[consideration_length + mov_avg_length]
    prev[1] = max_prices[consideration_length + mov_avg_length]
    prev[2] = min_prices[consideration_length + mov_avg_length]
    prev[3] = close_prices[consideration_length + mov_avg_length]

    for i in range(consideration_length + mov_avg_length, len(close_prices) - (skip_future_days + future_days + 1)):
        line = []
        line += open_prices[i - consideration_length:i]
        line += max_prices[i - consideration_length:i]
        line += min_prices[i - consideration_length:i]
        line += close_prices[i - consideration_length:i]
        line += moving_average_for_day[i - consideration_length:i]
        line += volumes[i - consideration_length:i]
        data.append(line)

        labels_at_day = []
        real_prices_at_day = []
        #classification
        '''for ft in range(1, future_days + 1):
            if close_prices[i + ft] > moving_average_for_day[i]:
                labels_at_day.append([1])
            else:
                labels_at_day.append([0])
            #labels_at_day.append(close_prices[i + ft])'''
        #simple regression
        '''for ft in range(1, future_days + 1):
            labels_at_day.append(moving_average_for_day[i + ft])
        labels.append(labels_at_day)'''
        #4 output regression
        for ft in range(skip_future_days + 1, skip_future_days + future_days + 1):
            current = np.zeros(4)
            rp_at_day = np.zeros(4)
            current[0] = (open_prices[i + ft] * 2/(1 + mov_avg_length)) + (prev[0] * (1 - 2/(1 + mov_avg_length)))
            current[1] = (max_prices[i + ft] * 2/(1 + mov_avg_length)) + (prev[1] * (1 - 2/(1 + mov_avg_length)))
            current[2] = (min_prices[i + ft] * 2/(1 + mov_avg_length)) + (prev[2] * (1 - 2/(1 + mov_avg_length)))
            current[3] = (close_prices[i + ft] * 2/(1 + mov_avg_length)) + (prev[3] * (1 - 2/(1 + mov_avg_length)))
            rp_at_day[0] = open_prices[i + ft]
            rp_at_day[1] = max_prices[i + ft]
            rp_at_day[2] = min_prices[i + ft]
            rp_at_day[3] = close_prices[i + ft]
            real_prices_at_day.append(rp_at_day)
            labels_at_day.append(current)
            prev = current
        real_prices.append(real_prices_at_day)
        labels.append(labels_at_day)


    for days in range(len(data)):
        base_price = data[days][input_size - 1]
        bases.append(labels[days][-1])
        base_volume = data[days][-1]
        for p in range(len(data[days]) - input_size):
            data[days][p] = (data[days][p] / base_price) * 5 - 5
            """if data[days][p] > 1:
                data[days][p] = 1
            if data[days][p] < -1:
                data[days][p] = -1"""
        for v in range(len(data[days]) - input_size, len(data[days])):
            relative_volume = data[days][v] / base_volume
            """if relative_volume < 0.1:
                relative_volume = 0.1
            if relative_volume > 2:
                relative_volume = 2"""
            data[days][v] = relative_volume - 1
        for d in range(len(labels[days])):
            labels[days][d] = (labels[days][d] / base_price - 1.0) * 20


    data = np.array(data)
    labels = np.array(labels)
    real_prices = np.array(real_prices)
    bases = np.array(bases)

    return data, labels, bases, real_prices


def open_stock_with_integral(name, days, input_size, integral_length, future_days, skip_future_days, skip):
    open_prices = []
    min_prices = []
    max_prices = []
    close_prices = []
    volumes = []
    bases = []
    integral_length = int(integral_length)
    consideration_length = input_size

    f = open(name, "r")
    linelist = f.readlines()
    for ln in range(days):
        d = -1 * days + ln - skip
        line = linelist[d].split(";")
        open_prices.append(line[1])
        max_prices.append(line[2])
        min_prices.append(line[3])
        close_prices.append(line[4])
        if len(line) == 6:
            volumes.append(line[5][:-2])
        else:
            volumes.append("1")

    for d in range(len(close_prices)):
        try:
            open_prices[d] = float(open_prices[d].replace(",", "."))
        except Exception as e:
            open_prices[d] = open_prices[d - 1]
        try:
            max_prices[d] = float(max_prices[d].replace(",", "."))
        except Exception as e:
            max_prices[d] = max_prices[d - 1]
        try:
            min_prices[d] = float(min_prices[d].replace(",", "."))
        except Exception as e:
            min_prices[d] = min_prices[d - 1]
        try:
            close_prices[d] = float(close_prices[d].replace(",", "."))
        except Exception as e:
            close_prices[d] = close_prices[d - 1]
    for d in range(len(volumes)):
        try:
            volumes[d] = float(volumes[d].replace(",", "."))
        except Exception as e:
            volumes[d] = 0.1

    open_filtered = []
    min_fitlered = []
    max_filtered = []
    close_filtered = []

    for d in range(integral_length):
        open_filtered.append(open_prices[d])
        min_fitlered.append(min_prices[d])
        max_filtered.append(max_prices[d])
        close_filtered.append(close_prices[d])

    for d in range(1 + integral_length, len(close_prices)):
        open_filtered.append(sum(open_prices[d - integral_length: d]) / integral_length)
        min_fitlered.append(sum(min_prices[d - integral_length: d]) / integral_length)
        max_filtered.append(sum(max_prices[d - integral_length: d]) / integral_length)
        close_filtered.append(sum(close_prices[d - integral_length: d]) / integral_length)

    data = []
    labels = []
    real_prices = []

    for i in range(consideration_length, len(close_prices) - (skip_future_days + future_days + 1)):
        line = []
        line += open_prices[i - consideration_length:i]
        line += max_prices[i - consideration_length:i]
        line += min_prices[i - consideration_length:i]
        line += close_prices[i - consideration_length:i]
        line += volumes[i - consideration_length:i]
        data.append(line)

        labels_at_timestep = []
        real_prices_at_timestep = []

        for ft in range(skip_future_days + 1, skip_future_days + 0*future_days + 1 + 1):
            labels_at_day = []
            real_prices_at_day = []
            current = np.zeros(4)
            rp_at_day = np.zeros(4)
            index = i + ft

            current[0] = open_filtered[index]
            current[1] = min_fitlered[index]
            current[2] = max_filtered[index]
            current[3] = close_filtered[index]
            rp_at_day[0] = open_prices[index]
            rp_at_day[1] = min_prices[index]
            rp_at_day[2] = max_prices[index]
            rp_at_day[3] = close_prices[index]

            labels_at_day.append(current)
            real_prices_at_day.append(rp_at_day)

            labels_at_timestep.append(labels_at_day)
            real_prices_at_timestep.append(real_prices_at_day)

        real_prices.append(real_prices_at_timestep)
        labels.append(labels_at_timestep)

    relative_volume_limit = 10
    price_scale_coeficient = 5
    label_scale_coeficient = 20

    final_data = []

    for days in range(0, len(data) - (future_days - 1), future_days):
        base_price = data[days][input_size - 1]
        bases.append(labels[days][-1])
        base_volume = data[days][-1]
        single_day = np.zeros((future_days, 5 * input_size))
        for ft in range(future_days):
            for p in range(len(data[days]) - input_size):
                single_day[ft][p] = (data[days][p] / base_price) * price_scale_coeficient - price_scale_coeficient
            for v in range(len(data[days]) - input_size, len(data[days])):
                relative_volume = data[days][v] / base_volume
                single_day[ft][v] = relative_volume - 1
                if single_day[ft][v] > relative_volume_limit:
                    single_day[ft][v] = relative_volume_limit
                if single_day[ft][v] < 1/relative_volume_limit:
                    single_day[ft][v] = 1/relative_volume_limit
        final_data.append(single_day)

        # for p in range(len(data[days]) - input_size):
        #     data[days][p] = (data[days][p] / base_price) * price_scale_coeficient - price_scale_coeficient
        # for v in range(len(data[days]) - input_size, len(data[days])):
        #     relative_volume = data[days][v] / base_volume
        #     data[days][v] = relative_volume - 1
        #     if data[days][v] > relative_volume_limit:
        #         data[days][v] = relative_volume_limit
        #     if data[days][v] < 1/relative_volume_limit:
        #         data[days][v] = 1/relative_volume_limit

        future = future_days
        for ft in range(future):
            if days + ft < len(labels):
                d = 0
                labels[days + ft][d][0] = (labels[days + ft][d][0] / base_price - 1.0) * label_scale_coeficient
                if any(labels[days + ft][d][0] > 1) or any(labels[days + ft][d][0] < -1):
                    for price in range(4):
                        if labels[days + ft][d][0][price] > 1:
                            labels[days + ft][d][0][price] = 1
                        if labels[days + ft][d][0][price] < -1:
                            labels[days + ft][d][0][price] = -1

    final_labels = []
    for day in range(0, len(labels), future_days):
        if day + ft < len(labels):
            single_labels = []
            for ft in range(future_days):
                single_labels.append(labels[day + ft])
            final_labels.append(single_labels)

    data = np.array(final_data)
    labels = np.array(final_labels)
    real_prices = np.array(real_prices)
    bases = np.array(bases)

    return data, labels, bases, real_prices

def write11b():
    prices = []

    name = "C:\\Users\\PZI004\\PycharmProjects\\untitled\\venv\\11BIT.csv"
    f = open(name, "r")
    linelist = f.readlines()
    f.close()

    name2 = "11b.txt"
    f = open(name2, "w+")
    for d in range(1000):
        price = linelist[d].split(";")
        price = price[4].replace(",", ".")
        prices.append(price)
    for p in prices:
        f.write(p + "\r\n")
    f.close()
