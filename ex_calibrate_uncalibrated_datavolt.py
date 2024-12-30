from main import *

run_list = [0, 1, 2, 11, 20, 30, 40, 50, 56]
board = 10
for run in run_list:
    calibrate_uncalibrated_datavolt(run, board)
