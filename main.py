import math
import random

import numpy as np
import matplotlib.pyplot as plt

from MFDFA import fgn
from hurst import compute_Hc

DATA_FILES = ['./SPY_DATA/2022-12-1 SPY.txt',
              './SPY_DATA/2022-12-2 SPY.txt',
              './SPY_DATA/2022-12-5 SPY.txt',
              './SPY_DATA/2022-12-6 SPY.txt',
              './SPY_DATA/2022-12-7 SPY.txt',
              './SPY_DATA/2022-12-8 SPY.txt',
              './SPY_DATA/2022-12-9 SPY.txt',
              './SPY_DATA/2022-12-12 SPY.txt',
              './SPY_DATA/2022-12-13 SPY.txt',
              './SPY_DATA/2022-12-14 SPY.txt',
              './SPY_DATA/2022-12-15 SPY.txt',
              './SPY_DATA/2022-12-16 SPY.txt',
              './SPY_DATA/2022-12-19 SPY.txt',
              './SPY_DATA/2022-12-20 SPY.txt',
              './SPY_DATA/2022-12-21 SPY.txt',
              './SPY_DATA/2022-12-22 SPY.txt',
              './SPY_DATA/2022-12-27 SPY.txt',
              './SPY_DATA/2022-12-28 SPY.txt',
              './SPY_DATA/2022-12-29 SPY.txt',
              './SPY_DATA/2022-12-30 SPY.txt',
              './SPY_DATA/2023-1-3 SPY.txt',
              './SPY_DATA/2023-1-4 SPY.txt',
              './SPY_DATA/2023-1-5 SPY.txt',
              './SPY_DATA/2023-1-6 SPY.txt',
              './SPY_DATA/2023-1-9 SPY.txt',
              './SPY_DATA/2023-1-10 SPY.txt',
              './SPY_DATA/2023-1-11 SPY.txt',
              './SPY_DATA/2023-1-12 SPY.txt',
              './SPY_DATA/2023-1-13 SPY.txt',
              './SPY_DATA/2023-1-17 SPY.txt',
              './SPY_DATA/2023-1-18 SPY.txt',
              './SPY_DATA/2023-1-19 SPY.txt',
              './SPY_DATA/2023-1-20 SPY.txt',
              './SPY_DATA/2023-1-23 SPY.txt',
              './SPY_DATA/2023-1-24 SPY.txt',
              './SPY_DATA/2023-1-25 SPY.txt',
              './SPY_DATA/2023-1-26 SPY.txt',
              './SPY_DATA/2023-1-27 SPY.txt',
              './SPY_DATA/2023-1-30 SPY.txt',
              './SPY_DATA/2023-1-31 SPY.txt',
              './SPY_DATA/2023-2-1 SPY.txt',
              './SPY_DATA/2023-2-2 SPY.txt',
              './SPY_DATA/2023-2-3 SPY.txt',
              './SPY_DATA/2023-6-2 SPY.txt',
              './SPY_DATA/2023-6-7 SPY.txt',
              './SPY_DATA/2023-6-8 SPY.txt',
              './SPY_DATA/2023-6-9 SPY.txt',
              './SPY_DATA/2023-6-12 SPY.txt',
              './SPY_DATA/2023-6-13 SPY.txt',
              './SPY_DATA/2023-6-14 SPY.txt',
              './SPY_DATA/2023-6-15 SPY.txt',
              './SPY_DATA/2023-6-16 SPY.txt',
              ]
# DATA_FILES = ['./SPY_DATA/2023-6-16 SPY.txt']

# Buy signals:
## if H > 0.5 and there was an increase in avg price in past X timesteps
## if H < 0.5 and there was a decrease to below the moving avg
# Sell signals:
## if H > 0.5 and there was an increase in avg price in the past X timesteps
## if H < 0.4 and there was an increase to above the moving avg
BACKTEST = True
NEAR_AVG = 10
FAR_AVG = 50
positions = {"SPY": 0}
money = 0
buys = []
sells = []


def get_hurst(seq):
    seq = np.array(seq)
    max_window_size = len(seq) // 4
    step_size = len(seq) // 64
    Ls = np.array(range(1, max_window_size, step_size))
    S = np.zeros(len(Ls))
    for i in range(len(Ls)):
        l = Ls[i]
        xs = seq[0:(len(seq) - l)] - seq[l:len(seq)]
        counts, b = np.histogram(xs)
        counts = counts[counts != 0]
        binsize = b[1] - b[0]
        P = counts/sum(counts)
        S[i] = -sum(P * np.log(P)) + math.log(binsize)
    Ls = np.log(Ls)

    fit_start = 0
    fit_end = len(Ls) // 4

    coeff = np.polyfit(Ls[fit_start:fit_end], S[fit_start: fit_end], 1)
    H = coeff[0]
    return H

def get_series(file):
    ret = []
    with open(file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        ret.append(float(line.split(',')[1]))
    return ret


def main(file):
    global money
    series = get_series(file)
    starting_money = 9000
    money = starting_money
    print(file)
    print(f"STARTING MONEY: {money}")
    N = 100
    hs = []
    near_avg = []
    far_avg = []
    buys.append([0, series[0]])
    sells.append([0, series[0]])
    for i in range(N):
        hs.append(0)
    for i in range(len(series) - N):
        curr_window = series[:i + N]
        H = get_hurst(curr_window)
        hs.append(H)
    for i in range(10):
        near_avg.append(series[i])
    for i in range(len(series) - 10):
        curr_window = series[i:i + 10]
        near_avg.append(sum(curr_window) / 10)

    for i in range(50):
        far_avg.append(series[i])
    for i in range(len(series) - 50):
        curr_window = series[i:i + 50]
        far_avg.append(sum(curr_window) / 50)
    for i in range(N, len(series)):
        make_decision(series[:i])
    money += positions["SPY"] * series[-1]
    positions["SPY"] = 0
    print(f"ENDING MONEY: {money}")
    print(f"DIFFERENCE: {money - starting_money}")
    print(f"BASELINE: {starting_money/series[100] * series[-1]}")
    print("--------------------------")

    fig, ax = plt.subplots()
    plt.title(file)
    ax.plot(series)
    ax.plot(near_avg, color='yellow')
    ax.plot(far_avg, color='orange')
    ax.scatter(*zip(*buys), color='green')
    ax.scatter(*zip(*sells), color='red')
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax2 = ax.twinx()
    ax2.plot(hs, color='purple')
    baseline = [0.5 for i in range(len(series))]
    ax2.plot(baseline, color='orange')
    ax2.set_ylabel("H")
    plt.show()

    buys.clear()
    sells.clear()
    return money - starting_money, starting_money / series[100] * series[-1] - starting_money


def buy(q, price, timestep):
    global money
    if BACKTEST:
        positions["SPY"] += q
        money -= q * price
        buys.append([timestep, price])
    # TODO add api interaction


def sell(q, price, timestep):
    global money
    if BACKTEST:
        positions["SPY"] -= q
        money += q * price
        sells.append([timestep, price])
    # TODO add api interaction


def make_decision(prices):
    far_avg = sum(prices[-FAR_AVG:])/FAR_AVG
    p_far_avg = sum(prices[-FAR_AVG - 1:-1])/FAR_AVG
    price_diff = prices[-1] - prices[-2]
    near_avg = sum(prices[-NEAR_AVG:])/NEAR_AVG
    h, c, data = compute_Hc(prices)
    if h > 0.6:
        if far_avg - p_far_avg > 0.01 and positions["SPY"] < 20:
            buy(1, prices[-1], len(prices) - 1)
        if far_avg - p_far_avg < -0.01 and positions["SPY"] > -20:
            sell(1, prices[-1], len(prices) - 1)
    if h < 0.45:
        if prices[-1] < near_avg and positions["SPY"] < 20:
            buy(1, prices[-1], len(prices) - 1)
        if prices[-1] > near_avg and positions["SPY"] > -20:
            sell(1, prices[-1], len(prices) - 1)


if __name__ == '__main__':
    tot = 0
    base = 0
    print(len(DATA_FILES))
    for file in DATA_FILES:
        t, b = main(file)
        tot += t
        base += b
    print(tot, base)

