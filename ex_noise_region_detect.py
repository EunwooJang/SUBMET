from main import *

run_list = [0, 1, 2, 11, 20, 30, 40, 50, 56, 101]
board = 10
for run in run_list:
    noise_region_detect(run, board)
