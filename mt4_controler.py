import omq
from pandas import DataFrame, Timestamp, Timedelta
import time
import numpy as np
import files2
import torch
import NN07_FX
from TraderClasses import Model, Set, WalletFX, OrderType, PositionType
from tsmoothie.smoother import *
from matplotlib import pyplot as plt


class MT4_CONTROLER:
    def __init__(self):
        self._zmq = omq.DWX_ZeroMQ_Connector()
        self.data_historical = {}
        self.actions = []
        self.second_counter = 0

        self.params = NN07_FX.Params()
        lstm_name = 'LSTM_FX_p2h_s2h'
        self.model_lstm = torch.load(lstm_name + ".pt")
        trader_name = 'fx_model_08_p2h_s2h_candles_large'
        self.model_trader = torch.load(trader_name + '.pt')
        self.walletFx = WalletFX('USDPLN.', 20, 10000)
        self.walletFx.live_acting = True
        self.walletFx.buy_coefficient = 1
        self.free_margin = 10000

        self.magic_long = 0
        self.magic_short = 1
        self.symbol = "USDPLN."
        self.lots = 0.01
        self.walletFx.lots = self.lots
        self.sl = 0
        self.tp = 0

        self.ma_length = self.params.future_days
        self.past_days = self.params.time_steps
        self.future_days = self.params.future_days
        self.scale_high = self.params.scale_high
        self.minutes_past_to_retreive = 60 * (self.ma_length + self.past_days)
        self.past_prices_considered_for_choices = 3
        self.candles = True

    def wait_second(self, seconds=3):
        self.second_counter += seconds
        time.sleep(seconds)

    def wait_period(self, period_to_wait):
        seconds_to_wait = 60 * period_to_wait - self.second_counter
        time.sleep(seconds_to_wait)
        self.second_counter = 0

    def minMaxScaler(self, value, min_v, max_v):
        return (value - min_v) / (max_v - min_v)

    def get_open_trades(self):
        self._zmq._DWX_MTX_GET_ALL_OPEN_TRADES_()
        self.current_open_trades = self._zmq._thread_data_output

    def get_free_margin(self):
        print('Free balance:')
        self._zmq._DWX_MTX_SEND_FREE_MARGIN_(self.symbol)
        self.wait_second()
        for i in self._zmq._thread_data_output:
            self.free_margin = i
        self.walletFx.balance = self.free_margin

    def close_long_trades(self):
        self.close_trades_by_magic(self.magic_long)
        self.report_action(Timestamp.now(), 'CLOSE_LONG')

    def close_short_trades(self):
        self.close_trades_by_magic(self.magic_short)
        self.report_action(Timestamp.now(), 'CLOSE_SHORT')

    def close_trades_by_magic(self, magic):
        self._zmq._DWX_MTX_CLOSE_TRADES_BY_MAGIC_(_magic=magic)

    def close_all_trades(self):
        self._zmq._DWX_MTX_CLOSE_ALL_TRADES_()
        self.report_action(Timestamp.now(), 'CLOSE_ALL')

    def get_historical_data(self):
        is_incorrect = True
        max_pass_counter = 10
        pass_counter = max_pass_counter
        while is_incorrect:
            end = Timestamp.now() - Timedelta(hours=0)
            t = (end - Timedelta(minutes=self.minutes_past_to_retreive + 60 * 60)).strftime('%Y.%m.%d %H:%M:00')
            print('Retreiving data for end time: ' + end.strftime('%Y.%m.%d %H:%M:00') + ', start time: ' + t)
            self._zmq._DWX_MTX_SEND_HIST_REQUEST_(_symbol=self.symbol,
                                                  _start=t,
                                                  _end=end,
                                                  _timeframe=1)
            self.wait_second()
            self.wait_second()
            length = len(self._zmq._History_DB[self.symbol + '_M1'])
            if length > 60 * (self.ma_length + self.past_days) or pass_counter == 0:
                is_incorrect = False
                print('Loaded historical data correctly')
                if pass_counter == 0:
                    missing_steps = 60 * (self.ma_length + self.past_days) - length
                    self.data_historical = self._zmq._History_DB
                    data = self.data_historical[self.symbol + '_M1']
                    clone = data[:missing_steps]
                    for i in range(missing_steps):
                        data.insert(i, clone[i])
                    print('Historical data loaded incorrectly {} times, '.format(max_pass_counter)
                          + str(missing_steps) + ' steps copied')
                else:
                    self._zmq._History_DB[self.symbol + '_M1'] = self._zmq._History_DB[self.symbol + '_M1'][:self.minutes_past_to_retreive + 60]
                    self.data_historical = self._zmq._History_DB
                    data = self.data_historical[self.symbol + '_M1'][-self.minutes_past_to_retreive:]

                print(self.data_historical['USDPLN._M1'])
            else:
                print(str(length) + ' is incorrect length, re-loading......')
                pass_counter -= 1

        data_length = len(data)
        prices_m1 = np.zeros((data_length, 4))
        for i in range(data_length):
            prices_m1[i, 0] = data[i]['open']
            prices_m1[i, 1] = data[i]['high']
            prices_m1[i, 2] = data[i]['low']
            prices_m1[i, 3] = data[i]['close']

        prices_h1 = np.zeros((self.ma_length + self.past_days, 4))
        for i, i_m1 in enumerate(range(0, prices_m1.shape[0], 60)):
            prices_h1[i, 0] = prices_m1[i_m1, 0]
            prices_h1[i, 1] = np.max(prices_m1[i_m1:i_m1+60, 1])
            prices_h1[i, 2] = np.min(prices_m1[i_m1:i_m1+60, 2])
            prices_h1[i, 3] = prices_m1[i_m1+60-1, 3]

        self.prices_actual = self.prepare_prices(prices_h1)

    def predict_prices(self):
        states = self.model_lstm.return_states(self.params.lstm_layers_num, self.params.lstm_bidirectional, 1, self.params.lstm_hidden_sizes)
        predictions, states = self.model_lstm(self.prices_actual, states)
        self.predictions = predictions.detach().numpy()

    def choice(self):
        if self.candles:
            past_prices = self.prices_actual[:, 0, -(4 * self.past_prices_considered_for_choices):].reshape(-1)
            trader_input = np.zeros((4 * (self.past_prices_considered_for_choices + self.future_days) + 1))
            trader_input[:4 * self.past_prices_considered_for_choices] = past_prices
            trader_input[4 * self.past_prices_considered_for_choices:-1] = self.predictions.reshape(-1)
        else:
            past_prices = self.prices_actual[:, 0, -(4 * self.past_prices_considered_for_choices) + 3::4].reshape(-1)
            trader_input = np.zeros((self.past_prices_considered_for_choices + self.future_days + 1))
            trader_input[:self.past_prices_considered_for_choices] = past_prices
            trader_input[self.past_prices_considered_for_choices:-1] = self.predictions.reshape(-1)[3::4]

        trader_input[-1] = self.minMaxScaler(self.base_price, 2.5, 4.5)

        self.choices = self.model_trader(torch.tensor(trader_input).type(dtype=torch.float32))
        self.current_prices = np.zeros(3)
        self.current_prices[0] = self.data_historical[self.symbol + '_M1'][-1]['high']
        self.current_prices[1] = self.data_historical[self.symbol + '_M1'][-1]['low']
        self.current_prices[2] = self.data_historical[self.symbol + '_M1'][-1]['close']

        print('Predicted prices:')
        print(self.predictions[:, :, :])
        print('Choices:')
        print(self.choices.detach().numpy())

    def takeActions(self):
        mt4.get_free_margin()
        self.walletFx.takeAction(self.choices, self.current_prices, Timestamp.now())

        for action, position, order in zip(self.walletFx._action_to_take, self.walletFx._position_to_take, self.walletFx._orders_to_open):
            if action == OrderType.Close:
                if position == PositionType.Long:
                    self.close_long_trades()
                    self.wait_second()
                if position == PositionType.Short:
                    self.close_short_trades()
                    self.wait_second()

        mt4.get_free_margin()
        self.walletFx.takeAction(self.choices, self.current_prices, Timestamp.now())

        for action, position, order in zip(self.walletFx._action_to_take, self.walletFx._position_to_take, self.walletFx._orders_to_open):
            if action == OrderType.Open:
                if position == PositionType.Long:
                    ulots, minilots = self.calculate_orders_lots(order)
                    if ulots > 0:
                        self.lots = 0.01
                        for o in range(ulots):
                            self.open_long_position()
                            self.wait_second()
                    if minilots > 0:
                        self.lots = 0.1
                        for o in range(minilots):
                            self.open_long_position()
                            self.wait_second()
                if position == PositionType.Short:
                    ulots, minilots = self.calculate_orders_lots(order)
                    if ulots > 0:
                        self.lots = 0.01
                        for o in range(ulots):
                            self.open_short_position()
                            self.wait_second()
                    if minilots > 0:
                        self.lots = 0.1
                        for o in range(minilots):
                            self.open_short_position()
                            self.wait_second()
        self.walletFx._action_to_take = []
        self.walletFx._position_to_take = []
        self.walletFx._orders_to_open = []

        print('Actions done, waiting......')

    ### RETURNS LIST OF ORDER SIZES AS: ['0.01 LOT', '0.1' LOT]
    def calculate_orders_lots(self, ulot_count):
        minilot_count = int(np.floor(ulot_count / 10))
        ulot_count = int(ulot_count - minilot_count * 10)
        return ulot_count, minilot_count

    def open_long_position(self):
        _my_trade = self._generate_default_order_dict()
        _my_trade['_type'] = self.magic_long
        _my_trade['_magic'] = self.magic_long
        self._zmq._DWX_MTX_NEW_TRADE_(_order=_my_trade)
        self.report_action(Timestamp.now(), 'OPEN_LONG')

    def open_short_position(self):
        _my_trade = self._generate_default_order_dict()
        _my_trade['_type'] = self.magic_short
        _my_trade['_magic'] = self.magic_short
        self._zmq._DWX_MTX_NEW_TRADE_(_order=_my_trade)
        self.report_action(Timestamp.now(), 'OPEN_SHORT')

    def _generate_default_order_dict(self):
        return ({'_action': 'OPEN',
                 '_type': 0,
                 '_symbol': self.symbol,
                 '_price': 0.0,
                 '_SL': self.sl,  # SL/TP in POINTS, not pips.
                 '_TP': self.tp,
                 '_comment': '',
                 '_lots': self.lots,
                 '_magic': 0,
                 '_ticket': 0})

    def report_action(self, time, action):
        msg = ({'time': time,
                'action': action})
        self.actions.append(msg)

    def prepare_prices(self, data):
        prices = np.zeros((1, self.future_days, 4 * self.past_days))

        _range = np.moveaxis(data, 0, -1)
        filtered_range = np.zeros(_range.shape)

        smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
        smoother.smooth(_range)
        filtered_range = smoother.smooth_data[:, self.ma_length:]

        base_price = filtered_range[-1, -1].reshape((-1, 1))
        self.base_price = base_price
        relative_prices_x = files2.price_relation(filtered_range, base_price)
        normalized_prices_x = files2.normalize_price(relative_prices_x, self.scale_high, 1, clip=False)
        x = normalized_prices_x.reshape(-1)

        # _data_filtered = np.zeros((data.shape[0], 4))
        # _sums_complementary = np.zeros((data.shape[0], 4))
        #
        # for i in range(data.shape[1]):
        #     _data_filtered[:, i], _sums_complementary[:, i] = files2.calculate_ma(self.ma_length, data[:, i])
        #
        # base_price = _data_filtered[-1, -1]
        # prices_relative = files2.price_relation(_data_filtered[-self.past_days:, :], base_price)
        # normalized_prices = files2.normalize_price(prices_relative, self.scale_high, 1, clip=False)
        # x = normalized_prices
        mean = np.mean(x)
        std = np.std(x)
        print('Actual mean, std: ' + str(np.round(mean, 4)) + ' ' + str(np.round(std, 4)))
        print('Model mean, std: ' + str(np.round(self.model_lstm.mean, 4)) + ' ' + str(np.round(self.model_lstm.std, 4)))

        x = (x - self.model_lstm.mean) / self.model_lstm.std

        for i in range(self.future_days):
            prices[0, i, :] = x.reshape(-1)

        return torch.tensor(prices.reshape(1, self.future_days, 4 * self.past_days)).type(torch.float32)


mt4 = MT4_CONTROLER()

while True:
    mt4.wait_second()
    mt4.get_historical_data()
    mt4.predict_prices()
    mt4.choice()
    #mt4.takeActions()
    #mt4.close_short_trades()
    #mt4.close_long_trades()
    mt4.wait_period(period_to_wait=120)

