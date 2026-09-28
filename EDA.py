import numpy
import matplotlib
import pandas as pd
import pickle as pkl
from matplotlib import pyplot as plt
from sim_sweep_v2 import *
params = []
performance_metrics = dict()

summary_df, timeseries = load_sweep(out_dir="sweep_results", tag="pu_pe_ph")

print(len(timeseries[(0, 0.0025, 0.0025, 0.0025)]['Times']))
print(len(timeseries[(0, 0.0025, 0.0025, 0.0025)]['Now_consecutive']))

# print(timeseries[(0, 0.0025, 0.0025, 0.0025)].keys())

# for key1 in timeseries.keys():
#     if [key1[1], key1[2], key1[3]] not in params:
#         plt.title('Parameters = (' + str(key1[1]) + ', ' + str(key1[2]) + ', ' + str(key1[3]) + ')')
#         params.append([key1[1], key1[2], key1[3]])
#         for key2 in timeseries.keys():
#             if key2[1] == key1[1] and key2[2] == key1[2] and key2[3] == key1[3]:
#                 plt.scatter(x = timeseries[key2]['Times'], y = timeseries[key2]['Now_consecutive'])
#         plt.xlim(left = 365, right = 1095)
#         plt.show()

# dict_keys formatted (sim, p_h, p_e, p_u)
# for key in timeseries.keys():
#     performance_metrics[key] =
#
# print(timeseries.keys())