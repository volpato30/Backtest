from sys import path
path.append('/work/rqiao/HFdata/mew-p')
path.append('/work/rqiao/HFdata/algorithms')
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
from mewp.simulate import MatchOptions
from mewp.simulate import TakerMatchMode
import numpy as np
import itertools
from algo_constant import ConstantAlgo
from mewp.data.item import Contract
from mewp.util.pair_trade_analysis import TradeAnalysis
from mewp.data.order import OrderMethod
from mewp.data.order import OrderType
from mewp.util.futures import get_best_pair
from mewp.simulate.runner import PairRunner
from mewp.simulate.wrapper import PairAlgoWrapper
from mewp.simulate.report import Report
import pickle
from joblib import Parallel, delayed
from mewp.util.futures import get_best_pair
import pandas as pd
MatchOptions.future_tick = True
MatchOptions.taker_mode = TakerMatchMode.BY_NEXT_TICK

def back_test(pair, date, param):
    tracker = TradeAnalysis(Contract(pair[0]))
    algo = { 'class': ConstantAlgo }
    algo['param'] = {'x': pair[0],
                     'y': pair[1],
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'bollinger': param[1],
                     'const': param[2],
                     'block': 100,
                     'tracker': tracker
                     }
    settings = { 'date': date,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo,
                 'singletick': False}
    runner = PairRunner(settings)
    runner.run()
    account = runner.account
    history = account.history.to_dataframe(account.items)
    score = float(history[['pnl']].iloc[-1])
    order_win = tracker.order_winning_ratio()
    order_profit = tracker.analyze_all_profit()[0]
    num_rounds = tracker.analyze_all_profit()[2]
    return score, order_win, order_profit, num_rounds

def run_simulation(param, date_list, product):
    pnl_list = []
    order_win_list = []
    order_profit_list = []
    num_rounds_list = []
    for date in date_list:
        date_pair = get_best_pair(date, product, DATA_PATH)
        if type(date_pair) != tuple:
            continue
        else:
            result = back_test(date_pair, date, param)
            pnl_list.append(result[0])
            order_win_list.append(result[1])
            order_profit_list.append(result[2])
            num_rounds_list.append(result[3])
    return pnl_list, order_win_list, order_profit_list, num_rounds_list


# In[32]:

date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-07-01','2016-07-31').tolist()]
roll_list = np.concatenate((np.arange(200,500,200), np.arange(500, 3500, 500),[5000, 8000])) #11 params
sd_list = np.concatenate((np.arange(0, 2.1, 0.5),[3,4,5,8]))#11 params
spread_list = np.concatenate((np.arange(0, 2.1, 0.5),[3,4,5,8]))#11 params
pars = list(itertools.product(roll_list, sd_list, spread_list))
product = 'ni'
trade_day_list = []
for date in date_list:
    date_pair = get_best_pair(date, product, DATA_PATH)
    if type(date_pair) != tuple:
        continue
    else:
        trade_day_list.append(date)


num_cores = 20
results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param, date_list, product) for param in pars)
keys = ['roll:{}_sd:{}_spreadcoef:{}'.format(*p) for p in pars]
pnl = [i[0] for i in results]
order_win = [i[1] for i in results]
order_profit = [i[2] for i in results]
num_rounds = [i[3] for i in results]
pnl_dict = dict(zip(keys, pnl))
pnl_result = pd.DataFrame(pnl_dict)
pnl_result.index = trade_day_list
pnl_result.to_csv('./out/{}_benchmark_daily_pnl.csv'.format(product))
order_win_dict = dict(zip(keys, order_win))
order_win_result = pd.DataFrame(order_win_dict)
order_win_result.index = trade_day_list
order_win_result.to_csv('./out/{}_benchmark_daily_order_win.csv'.format(product))
order_profit_dict = dict(zip(keys, order_profit))
order_profit_result = pd.DataFrame(order_profit_dict)
order_profit_result.index = trade_day_list
order_profit_result.to_csv('./out/{}_benchmark_daily_order_profit.csv'.format(product))
num_rounds_dict = dict(zip(keys, num_rounds))
num_rounds_result = pd.DataFrame(num_rounds_dict)
num_rounds_result.index = trade_day_list
num_rounds_result.to_csv('./out/{}_benchmark_daily_num_rounds.csv'.format(product))
