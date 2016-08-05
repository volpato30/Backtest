import sys
from sys import path
import numpy as np
import pandas as pd
import random
path.append('/work/rqiao/HFdata/cython_mew-p')
path.append('/work/rqiao/HFdata/algorithms')
DATA_PATH = '/work/rqiao/HFdata/dockfuture'
from algo_const_stopwin_guard import ConstStopWinGuardAlgo
from mewp.data.item import Contract
from mewp.util.pair_trade_analysis import TradeAnalysis
from mewp.simulate.runner import PairRunner
import itertools
import pickle
from joblib import Parallel, delayed
from mewp.simulate import MatchOptions
from mewp.simulate import TakerMatchMode
from mewp.simulate.pair_exe import PairExeBase
from mewp.data.order import OrderMethod
from mewp.data.order import OrderType
from mewp.simulate.report import Report
from mewp.simulate.pair_exe import PairExePlusTick

str_rate = sys.argv[1]
RATE = float(str_rate)
MatchOptions.taker_mode = TakerMatchMode.BY_RATE
MatchOptions.taker_rate = dict()
MatchOptions.taker_rate[Contract("ni1609").cid] = RATE
MatchOptions.taker_rate[Contract("ni1701").cid] = RATE
seed_list = np.arange(10)
product = 'ni'
market = 'shfe'

dates = [str(x).split(' ')[0] for x in pd.date_range('2016-07-01','2016-07-31').tolist()]

def run_simulation(param, date_list):
    algo = { 'class': ConstStopWinGuardAlgo }
    algo['param'] = {'x': 'ni1609',
                     'y': 'ni1701',
                     'a': 1,
                     'b': 0,
                     'rolling': param[0],
                     'bollinger': 2,
                     'const': param[1],
                     'stop_win':param[2],
                     'block': 100,
                     'tracker': None
                    }
    settings = { 'date': date_list,
                 'path': DATA_PATH,
                 'tickset': 'top',
                 'algo': algo,
                 'singletick': False}
    settings['exe'] = PairExePlusTick(2)
    runner = PairRunner(settings)
    runner.run()
    report = Report(runner)
    temp = report.get_daily_pnl()
    pnl_list = list(temp.daily_pnl)
    return pnl_list
#
date_list = [str(x).split(' ')[0] for x in pd.date_range('2016-07-01','2016-07-31').tolist()]
roll_list = np.concatenate((np.arange(200,500,200), np.arange(500, 3500, 500),[5000, 8000, 15000])) #12 params
const_list = np.concatenate((np.arange(1.25, 2.5, 0.25),np.arange(2.5, 5.5, 0.5),np.arange(6,15,2)))#15 params
stopwin_list = np.concatenate((np.arange(2, 13),[15,20,100]))#14 params

pars = list(itertools.product(roll_list, const_list, stopwin_list))
num_cores = 20
trade_day_list = []
for date in date_list:
    date_pair = get_best_pair(date, product, DATA_PATH)
    if type(date_pair) != tuple:
        continue
    else:
        trade_day_list.append(date)
for seed in seed_list:
    random.seed(seed)
    results = Parallel(n_jobs=num_cores)(delayed(run_simulation)(param,\
        date_list) for param in pars)
    keys = ['roll:{}_const:{}_stopwin:{}'.format(*p) for p in pars]
    pnl_dict = dict(zip(keys, results))
    pnl_result = pd.DataFrame(pnl_dict)
    pnl_result.index = trade_day_list
    pnl_result.to_csv('./out/ni_stopwin_daily_pnl_slipery_point:{}_seed:{}.csv'.format(str_rate,seed))
