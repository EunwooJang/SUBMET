from main import *

run_list = [0, 40]
board = 10

# noise sample 생성하기
extract_noise_sample(run_list, board)

# noise sample 불러오기
noise_sample = np.load(f"files/noise sample/noise_sample_{board}.npy")

# 채널별 noise sample 그래프로 그리기
for ch in range(16):
    array = noise_sample[ch]
    plot_single_array(array,
                      f"Ch. {ch} Noise sample", "ADC", "TDC",
                      ylim=[0, 3800], mode="line", datatype="ADC", points=None, show=1,
                      save=0, path=None)
