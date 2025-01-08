from main import *

run_list = [0, 1, 2, 11, 20, 30, 40, 50, 56, 101]
board = 10

# 오로지 노이즈 구간들을 cell을 기준으로 정렬해서 그리기 참고로 아닌 부분은 값을 4095으로 채우기

header_dict = {run: np.load(f"files/header/header_{run}_{board}.npy") for run in run_list}
datavolt_dict = {run: np.load(f"files/datavolt/datavolt_{run}_{board}.npy") for run in run_list}
noise_dict = \
    {run: np.load(f"files/noise detection result/noise_detection_result_{run}_{board}.npy") for run in run_list}

ord_mean_std = datavolt_mean_std(0, board)

all_noise_mean_std = []
ch_noise_mean_max = []

for channel in range(16):
    ch_noise_mean_std = []

    for run in run_list:
        print(run)
        for result in noise_dict[run]:
            ch, event, shift, start, end, types = result
            if end != 0 and ch == channel:
                single_datavolt = datavolt_dict[run][ch, event, :]

                # single_datavolt에서 start:end+1에 해당하는 영역을 제외하고 나머지 부분을 4095로 설정
                if start > 0:
                    single_datavolt[:start] = 4095

                if end < single_datavolt.size - 1:
                    single_datavolt[end - 1:] = 4095

                # 2600 이하의 값들과 주변 ±5 값들을 4095로 설정
                indices = np.where(single_datavolt <= 2600)[0]
                for index in indices:
                    start_idx = max(0, index - 5)
                    end_idx = min(len(single_datavolt), index + 5 + 1)
                    single_datavolt[start_idx:end_idx] = 4095

                count = sum(1 for x in single_datavolt if x != 4095)

                if count > 100:
                    rotate = single_datavolt_cell_align(single_datavolt, header_dict[run][ch, event])
                    plot_single_array(rotate, title=f"Board.{board} Run.{run} Event.{event} Type{types}",
                                      datatype="ADC", save=1,
                                      ylim=[2000, 3800],
                                      path=f"files/images/noise detection only noise with cell align/ch{ch}/")

                    # 4095가 아닌 값들의 인덱스 추출
                    valid_indices = np.where(rotate != 4095)[0]

                    # 연속 구간 추출
                    gaps = np.diff(valid_indices)
                    split_points = np.where(gaps > 1)[0]
                    split_ranges = np.split(valid_indices, split_points + 1)

                    # 길이가 100 이상인 구간 처리
                    for valid_range in split_ranges:
                        if len(valid_range) >= 100:
                            segment = rotate[valid_range]
                            mean = np.mean(segment)
                            std = np.std(segment)
                            ch_noise_mean_std.append([mean, std])
                            all_noise_mean_std.append([channel, mean, std])

    # ch_noise_mean_std에 있는 mean, std 값을 점 플롯으로 시각화
    if ch_noise_mean_std:
        ch_noise_mean_std = np.array(ch_noise_mean_std)
        means = ch_noise_mean_std[:, 0]
        stds = ch_noise_mean_std[:, 1]
        ch_noise_mean_max.append(ord_mean_std[channel][0])

        plt.figure(figsize=(12, 6))
        plt.scatter(means, stds, s=10, marker='o', alpha=0.5, color='blue')
        plt.xlabel("Mean (ADC)")
        plt.ylabel("Standard Deviation (ADC)")
        plt.title(f"Ch.{channel} Noise STD vs Mean")
        plt.ylim(0, 150)
        plt.xlim(ord_mean_std[channel][0] - 1000, ord_mean_std[channel][0] + 30)  # 1024 = 10비트 언저리의 범위안에 다 들어있네?
        plt.grid(True)
        plt.savefig(f"files/images/noise std vs mean/Ch_{channel}_noise_std_vs_mean.png", dpi=300)
        # plt.show()
        plt.close()

plt.figure(figsize=(12, 6))

for channel in range(16):
    # 해당 채널의 mean과 std만 필터링
    channel_data = [data for data in all_noise_mean_std if data[0] == channel]
    if channel_data:
        channel_data = np.array(channel_data)
        means = channel_data[:, 1] - ch_noise_mean_max[channel]  # mean에서 해당 채널의 max 값 빼기
        stds = channel_data[:, 2]

        # 채널별 데이터 추가
        plt.scatter(means, stds, s=10, marker='o', alpha=0.1, color='blue')

# 그래프 설정
plt.xlabel("Mean (ADC) (Shifted by Max)")
plt.ylabel("Standard Deviation (ADC)")
plt.title("All Channels Noise STD vs Mean")
plt.xlim(-1000, 30)
plt.ylim(0, 150)
plt.grid(True)
plt.savefig("files/images/noise std vs mean/All_channels_noise_std_vs_mean.png", dpi=300)
# plt.show()
plt.close()
