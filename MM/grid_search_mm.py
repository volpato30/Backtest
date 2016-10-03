from sys import path
path.append('/work/rqiao/HFdata/mew-p')
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
from mewp.simulate.wrapper import MMWrapper
from mewp.simulate.runner import SingleRunner
from mewp.model.ladder_cython import LadderDual
from mewp.util.clock import Clock
from mewp.math.simple import SimpleMoving
from mewp.util.trade_period import Period
from mewp.util.mm import MMRecorder
from mewp.util.mm import TrendManager
from mewp.util.mm import TrendFinder
from mewp.util.mm import TrendState
from mewp.simulate.report import Report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from enum import Enum
from joblib import Parallel, delayed
import pickle

class VolatilityState(Enum):
    NORMAL = 1
    MEDIUM = 2
    LARGE = 3


class VolatilityFinder(object):
    def __init__(self, ma_diff_length, trigger_diff):
        self.ma_diff_length = ma_diff_length
        self.ma_diff = []
        self.extend_period = 10 * 60 * 1000
        self.trigger_diff = trigger_diff
        self.state = VolatilityState.NORMAL

    def checkpoint(self, ma_diff):
        self.ma_diff.append(ma_diff)
        if len(self.ma_diff) > self.ma_diff_length:
            self.ma_diff.pop(0)
        if self.state == VolatilityState.LARGE:
            if np.max(self.ma_diff) <= 6 and np.min(self.ma_diff) >= -6:
                self.state = VolatilityState.NORMAL
        elif self.state == VolatilityState.NORMAL:
            if abs(ma_diff) > self.trigger_diff:
                self.state = VolatilityState.LARGE
#
class MyTrendFinder(TrendFinder):
    def __init__(self):
        self.price_diff = SimpleMoving(size=20)
        self.short = SimpleMoving(size=10)
        self.long = SimpleMoving(size=8000)

        self.extend_period = 10 * 60 * 1000
        self.trigger_diff = 0.8
        self.extend_diff = 0.5
        self.volume_diff = 400
        self.volume_roll = []
        self.pd_roll = []
        self.last_v = 0
        self.last_p = 0

    # per tick checkpoint, for updating internal states
    def checkpoint(self, info):
        mid = (info.ask_1_price + info.bid_1_price)*0.5
        if self.last_p == 0:
            self.last_p = mid
        pd = abs(mid-self.last_p)
        self.short.add(mid)
        self.long.add(mid)
        self.price_diff.add(pd)
        self.volume_roll.append(info.volume - self.last_v)
        self.pd_roll.append(self.price_diff.mean)
        self.last_v = info.volume
        self.last_p = mid

    # check if should start
    def do_start(self, info):
        if self.volume_roll[-1] > self.volume_diff:
            return TrendState.UP_OR_DOWN

        if self.short.mean - self.long.mean > self.trigger_diff:
            return TrendState.UP

        if self.long.mean - self.short.mean > self.trigger_diff:
            return TrendState.DOWN

        return TrendState.NO

    # check if should extend
    def do_extend(self, info, curend):
        if self.volume_roll[-1] > self.volume_diff:
            return Clock.timestamp + self.extend_period
        if abs(self.short.mean - self.long.mean) > self.extend_diff:
            return Clock.timestamp + self.extend_period
        return curend
#
class MyMM(MMWrapper):

    # ------------------- wrapper call backs ---------------
    def param_updated(self):
        # make sure parent updates its param
        super(MyMM, self).param_updated()

        self.last_pnl = 0
        self.trend_manager = TrendManager(finder = MyTrendFinder())
        self.record = MMRecorder(self.item, self.account)
        self.volatility_finder = VolatilityFinder(self.param['ma_diff_length'], self.param['trigger_diff'])
        self.inv = self.param['inv_coef'] * self.item.symbol.min_ticksize
        self.chunk = self.param['chunk']

    def on_periodend(self, date, period, old_period):
        if period == Period.night or period == Period.first or period == Period.third:
            self.ladder.stop()
            self.tick_count = 0

    def on_daystart(self, date, info):
        self.day_setup()

    def on_dayend(self, date, info):
        self.trend_manager.end()
        self.ladder.stop()
        self.account.clear()

    def on_order_close(self, order):
        #print 'on close {} {}'.format(order.status, order.otype)
        pass

    def pause_for_n_ticks(self,n):
        self.pause_count = n

    def on_tick(self, multiple, contract, info):
        self.tick_count += 1
        self.record.checkpoint(info)
        self.trend_manager.checkpoint(info)
        self.mid_roll.add(self.record.last_mid)
        self.volatility_finder.checkpoint((self.record.last_mid-self.mid_roll.mean)/self.item.symbol.min_ticksize)
        #ma_diff.append((self.record.last_mid-self.mid_roll.mean)/0.05)

        if self.tick_count < self.block:
            return

        # pause
        if self.pause_count > 0:
            self.pause_count -= 1
            return

#         #  trending
#         if self.trend_manager.is_trending():
#             if self.ladder.is_running():
#                 self.ladder.stop()
#             return

        # check volatility state:
        if self.volatility_finder.state == VolatilityState.LARGE:
            if self.ladder.is_running():
                self.ladder.stop()
            return

        # check current mid price
        if abs(self.record.last_mid-self.mid_roll.mean) > self.spread * self.item.symbol.min_ticksize:
            return
            #self.pause_for_n_ticks(1000)


        #check position, stop bid or ask side



        # if not LARGE volatility
        if not self.ladder.is_running():
            self.ladder.set_mid(self.mid_roll.mean, self.chunk)
            self.ladder.start()
        else:
            mid = self.mid_roll.mean - self.inv * self.position()
            self.ladder.refresh(mid, self.chunk)

    def day_setup(self):
        self.mid_roll = SimpleMoving(size=500)
        self.spread = self.param['spread']
        self.ladder = LadderDual(self.item, qty=1, account=self.account, \
                                 spread=self.spread, gap=self.param['gap'], depth=3)
        self.tick_count = 0
        self.block = 1000
        self.pause_count = 0

def run_simulation(params):
    date = '2015-01-01'
    dateend = '2015-05-01'
    ma_diff = []
    dates = [str(x).split(' ')[0] for x in pd.date_range(date, dateend).tolist()]
    algo = { 'class': MyMM }
    temp = {'item': 'au1506'}
    temp['ma_diff_length'] = params[0]
    temp['trigger_diff'] = params[1]
    temp['ma_window'] = params[2]
    temp['spread'] = params[3]
    temp['inv_coef'] = params[4]
    temp['chunk'] = params[5]
    temp['gap'] = params[6]
    algo['param'] = temp
    settings = { 'date': dates, 'algo': algo, 'tickset': 'top', 'verbose' : True,
                     'path': DATA_PATH }
    runner = SingleRunner(settings)
    runner.run()
    report = Report(runner)
    pnl = report.get_final_pnl()
    sharp_ratio = report.get_sharpie_ratio()
    return pnl, sharp_ratio

#algo['param'] = {'item': 'au1612', 'ma_diff_length': 1000, 'trigger_diff': 12, 'ma_window': 500, 'spread': 4, 'inv_coef': 2, 'chunk': 3, 'gap': 3}
# A Search space with all the combinations over which the function will be minimized

ma_diff_length_list = np.arange(500,2000,500) # 3 params
trigger_diff_list = np.arange(6,21,3) # 5 params
ma_window_list = np.arange(500,2000,500) # 3 params
spread_list = np.arange(2,7,1) # 5 params
inv_coef_list = np.arange(1,5,1) # 4 params
chunk_list = np.arange(2,5,1) # 3 params
gap_list = np.arange(2,5,1) # 3 params
# in total 8100 param combinations
pars = list(itertools.product(ma_diff_length_list, trigger_diff_list, \
        ma_window_list, spread_list, inv_coef_list, chunk_list, gap_list))
num_cores = 32
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(params) \
        for params in pars)
with open('results.p', 'wb') as f:
    pickle.dump(results,f)
pnl = [i[0] for i in results]
sharp_ratio = [i[1] for i in results]
df = {'total_pnl': pnl, 'sharp_ratio': sharp_ratio}
df = pd.DataFrame(df)
df.index = pars
df.to_csv('./au_backtest.csv')
