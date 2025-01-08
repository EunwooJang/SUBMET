from main import *

run = 0
board = 10


# 단일 run의 이벤트들의 노이즈 구간 판단결과를 plt로 나타냄 (calibration_off = 1 이라면 uncalibrated 상태로 되돌림)
plot_noise_detection_result(run, board, 1, 0, 1, 0)
