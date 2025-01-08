# 기본적인 라이브러리
import numpy as np
import os

# 연산 속도 향상용 라이브러리
from numba import njit

# 그림 그리기용 라이브러리
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# 파일 디렉토리 생성
def directory_generator(basepath, structure):
    for folder, subfolders in structure.items():
        dir_path = os.path.join(basepath, folder)
        os.makedirs(dir_path, exist_ok=True)  # 디렉토리 생성
        if isinstance(subfolders, dict):  # 하위 디렉토리가 있는 경우 재귀적으로 처리
            directory_generator(dir_path, subfolders)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                                Raw data processing

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# raw 파일의 datavolt위치의 16비트를 그대로 읽기
def raw_to_datavolt_binary_read(run, board, ch, event):

    print(f"Ch.{ch} Event.{event}:")

    # 파일 경로 설정
    file_path = f"files/raw/raw_{run}_{board}.dat"

    # 파일 전체를 읽기
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # 파일 크기를 기반으로 이벤트 크기 계산
    file_size_in_bytes = os.path.getsize(file_path)
    event_size = int(file_size_in_bytes * 8 / (16 * 16 * 4096))

    if event > event_size:
        raise ValueError("Event is out of range")

    for tdc in range(4094):
        # 이벤트별 기본 오프셋 계산
        base_offset = 4096 * (16 * 2) * event + 2 * ch
        offsets = tdc * (16 * 2) + base_offset + (16 * 2) * 2

        # 두 바이트 데이터를 읽고 12비트로 변환
        data_high = file_data[offsets]  # 상위 바이트
        data_low = file_data[offsets + 1]  # 하위 바이트

        # 두 바이트 데이터를 병합하여 16비트로 변환
        data_pairs = (data_low.astype(np.uint16) << 8) | data_high.astype(np.uint16)

        # data_pairs를 바이너리 형식으로 출력
        print(bin(data_pairs)[2:].zfill(16), data_pairs, tdc)
        # print(bin(data_pairs & 0xFFF)[2:].zfill(12), data_pairs & 0xFFF, tdc)
    return


# raw 파일을 datavolt array로 변환
def raw_to_datavolt(file_path):
    # 파일 전체를 읽기
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # 파일 크기를 기반으로 이벤트 크기 계산
    file_size_in_bytes = os.path.getsize(file_path)
    event_size = int(file_size_in_bytes * 8 / (16 * 16 * 4096))

    # 배열 크기 및 초기화
    dv = np.full((16, event_size, 4094), -1, dtype=np.uint16)  # 0 ~ 4095이기에 12bit < 16bit

    # 데이터 처리
    for channel in range(16):
        for event in range(event_size):
            # 이벤트별 기본 오프셋 계산
            base_offset = 4096 * (16 * 2) * event + 2 * channel
            offsets = np.arange(4094) * (16 * 2) + base_offset + (16 * 2) * 2
            # 두 바이트 데이터를 읽고 12비트로 변환
            data_high = file_data[offsets]  # 상위 바이트
            data_low = file_data[offsets + 1]  # 하위 바이트
            data_pairs = (data_low.astype(np.uint16) << 8) | data_high.astype(np.uint16)  # 16비트 병합
            dv[channel, event] = data_pairs & 0xFFF

    return dv


# raw 파일을 datavolt array로 변환 및 저장 np.int16
def save_raw_to_datavolt(run, board):
    file_path = f"files/raw/raw_{run}_{board}.dat"
    # 파일에서 run, board 넘버? 로 구분된 것 같아 이를 파일 이름에 표기
    datavolt = raw_to_datavolt(file_path)
    np.save(f"files/datavolt/datavolt_{run}_{board}.npy", datavolt)
    print(f"datavolt {run} {board} file saved as files/datavolt/datavolt_{run}_{board}.npy")


# datavolt 불러오가
def load_datavolt(run, board):
    datvolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    return datvolt


# raw 파일의 헤더에서 event end cell number에 대응되는 정보를 추출
def raw_to_header(file_path):
    # 파일 전체를 읽기
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # 파일 크기를 기반으로 이벤트 크기 계산
    file_size_in_bytes = os.path.getsize(file_path)
    event_size = int(file_size_in_bytes * 8 / (16 * 16 * 4096))  # 16채널, 4096 데이터 포인트 가정

    # 배열 초기화
    header_info = np.full((16, event_size), -1, dtype=np.uint16)  # 16비트 저장

    # 헤더 위치 정보
    header_pos = [32, 32, 34, 34, 36, 36, 38, 38, 40, 40, 42, 42, 44, 44, 46, 46]

    # 데이터 처리
    for channel in range(16):
        for i in range(event_size):
            # 시작 위치 계산
            start_position = 32 * 4096 * i + header_pos[channel]
            data_high = file_data[start_position]
            data_low = file_data[start_position + 1]
            data_pairs = (data_low.astype(np.uint16) << 8) | data_high.astype(np.uint16)
            # 16비트 데이터 생성
            header_info[channel, i] = data_pairs

    return header_info


# raw 파일의 헤더에서 event end cell number에 대응되는 정보를 추출하여 저장 np.int16
def save_raw_to_header(run, board):
    file_path = f"files/raw/raw_{run}_{board}.dat"
    # 파일에서 run, board 넘버? 로 구분된 것 같아 이를 파일 이름에 표기
    header = raw_to_header(file_path)
    np.save(f"files/header/header_{run}_{board}.npy", header)
    print(f"header {run} {board} file saved as files/header/header_{run}_{board}.npy")


# datavolt 불러오가
def load_header(run, board):
    header = np.load(f"files/header/header_{run}_{board}.npy")
    return header


# raw 파일에서 나머지 헤더를 이것은 추출은 아니고 단순하게 한번에 plt로 나타내기
def plot_extra_header(run, board, pointss=None):
    file_path = f"files/raw/raw_{run}_{board}.dat"

    # 파일 전체를 읽기
    with open(file_path, 'rb') as file:
        file_data = np.frombuffer(file.read(), dtype=np.uint8)

    # 파일 크기를 기반으로 이벤트 크기 계산
    # file_size_in_bytes = os.path.getsize(file_path)
    # event_size = int(file_size_in_bytes * 8 / (16 * 16 * 4096))  # 16채널, 4096 데이터 포인트 가정
    # extra_header_pos = [32 ,34, 36, 38]
    extra_header_pos = list(range(0, 64, 2))
    for header_pos in extra_header_pos:

        header_info = np.full(2000, -1, dtype=np.uint16)

        for i in range(1, 2001):
            # 시작 위치 계산
            start_position = 32 * 4096 * i + header_pos
            data_high = file_data[start_position]
            data_low = file_data[start_position + 1]
            data_pairs = (data_low.astype(np.uint16) << 8) | data_high.astype(np.uint16)
            header_info[i - 1] = data_pairs

        plot_single_array(header_info, f"Extra header Pos.{header_pos // 2} Run.{run} Board.{board}",
                          "Event", "Header Value",
                          ylim=[np.min(header_info) - 10, np.max(header_info) + 10],
                          mode="line", datatype="else",
                          points=pointss, span=None,
                          show=0, save=1,
                          path=f"files/images/extra header/")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                                Calibration info

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# PMT 신호 datavolt, header에서 각 cell의 calibration_info 추출 및 저장. 여러 개의 run을 사용. np.int16
def extract_calibration_info(run_list, board):

    header_dict = {run: np.load(f"files/header/header_{run}_{board}.npy") for run in run_list}
    datavolt_dict = {run: np.load(f"files/datavolt/datavolt_{run}_{board}.npy") for run in run_list}

    calibration_info_candidates = [[[] for _ in range(4096)] for _ in range(16)]  # 16 x 4096 개의 빈 리스트들 집합
    value_count = np.zeros((16, 4096), dtype=np.int16)

    terminate = False  # 루프 종료 플래그 추가

    for run in run_list:
        if terminate:
            break  # run 루프 탈출
        print(f"Processing Run.{run}...")
        for ch in range(16):
            for event in range(len(datavolt_dict[run][0])):
                sing_event = datavolt_dict[run][ch][event]

                # 이벤트 내 값이 300 이하인지 확인
                if np.any(sing_event <= 300):  # 300 이하의 값이 하나라도 있는 경우에만 진행
                    shift = header_dict[run][ch][event]

                    # sing_event에서 값이 300 이하인 위치 및 값 추출
                    below_300_indices = [i for i, value in enumerate(sing_event) if value < 300]
                    below_300_indices_value = [sing_event[i] for i in below_300_indices]

                    # 인덱스를 shift 기준으로 조정
                    adjusted_indices = [
                        (shift - (4093 - indice) + 4096) % 4096 for indice in below_300_indices
                    ]
                    # calibration info 후보 리스트에 추가
                    if adjusted_indices:
                        for i in range(len(adjusted_indices)):
                            counts = value_count[ch, adjusted_indices[i]]
                            if counts < 50:
                                calibration_info_candidates[ch][adjusted_indices[i]].append(below_300_indices_value[i])
                                value_count[ch, adjusted_indices[i]] += 1

            # 3개 이상 값이 있는 리스트의 개수를 세고, 전체 칸 수와 함께 출력
            cells_with_3_or_more = sum(
                1 for ch_list in calibration_info_candidates for lst in ch_list if len(lst) >= 3
            )
            print(f"{cells_with_3_or_more} / {65536} cells have 3 or more values.")

            # 3개 이상인 칸의 개수가 총 칸 개수와 동일한 경우 조기 종료
            if cells_with_3_or_more == 65536:
                print("Enough calibration_info_candidates calculated")
                terminate = True  # 플래그 설정
                break  # ch 루프 탈출

    # 찾지 못한 cell 위치가 있는지 확인
    unique_positions = []
    for ch in range(16):
        for cell in range(4096):
            count = value_count[ch, cell]
            values = calibration_info_candidates[ch][cell]
            unique_values = set(values[:count])
            if len(unique_values) == count:  # 모든 값이 고유하면
                unique_positions.append((ch, cell))
    if unique_positions:
        print("Uncalculated cell positions:")
        for ch, cell in unique_positions:
            print(f"Channel {ch}, Cell {cell}")
        raise ValueError("Uncalculated cell positions existed")

    # 모든 16 * 4096개의 cell 위치에 사용되는 calibration info가 수집되었으므로 아래의 코드 실행
    calibration_info = np.full((16, 4096), -1, dtype=np.int16)  # 초기값은 -1로 설정
    for ch in range(16):
        for cell in range(4096):
            values = calibration_info_candidates[ch][cell]  # 현재 cell의 값 리스트
            if len(values) >= 2:  # 값이 2개 이상인 경우
                unique, counts = np.unique(values, return_counts=True)  # 고유 값과 개수 계산
                if len(unique) == 1:  # 모든 값이 동일한 경우
                    calibration_info[ch][cell] = unique[0]
                elif len(unique) > 1:  # 여러 값이 존재하는 경우
                    most_frequent_value = unique[np.argmax(counts)]  # 가장 많이 나타나는 값
                    if counts[np.argmax(counts)] > 1:  # 가장 많이 나타난 값이 2번 이상 나타난 경우만 저장
                        calibration_info[ch][cell] = most_frequent_value

    np.save(f"files/calibration info/calibration_info_{board}.npy", calibration_info)
    print(f"calibration info {board} file saved as files/calibration info/calibration_info_{board}.npy")


# 단일 datavolt event를 calibration 이전 값으로 바꿔서 내보내기 np.int16
def revert_single_event_to_uncalibrated(single_event_datavolt,
                                        single_event_header,
                                        ch_calibration_info):

    shift_calibration_info = np.roll(ch_calibration_info, 4095 - single_event_header)
    sliced_calibration_info = shift_calibration_info[2:]
    uncalbrated_datavolt = single_event_datavolt - sliced_calibration_info
    return uncalbrated_datavolt.astype(np.int16)


# datavlot를 uncalibrated datavolt로 새로 생성 및 저장 np.int16
def revert_datavolt_to_uncalibrated(run, board):
    header = np.load(f"files/header/header_{run}_{board}.npy")
    datavolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    calibration_info = np.load(f"files/calibration info/calibration_info_{board}.npy")

    event_size = len(datavolt[0])

    uncalibrated_datavolt = np.full((16, event_size, 4094), -1, dtype=np.uint16)
    for ch in range(16):
        for event in range(event_size):
            uncalibrated_datavolt[ch, event, :] = revert_single_event_to_uncalibrated(datavolt[ch, event, :],
                                                                                      header[ch, event],
                                                                                      calibration_info[ch])

    np.save(f"files/uncalibrated datavolt/uncalibrated_datavolt_{run}_{board}.npy", uncalibrated_datavolt)
    print(f"uncalibrated datavolt {run} {board} file "
          f"saved as files/uncalibrated datavolt/uncalibrated_datavolt_{run}_{board}.npy")


# uncalibrated datvolt를 calibration info를 이용해서 calibrate 및 저장 np.int16
def calibrate_uncalibrated_datavolt(run, board):
    header = np.load(f"files/header/header_{run}_{board}.npy")
    uncalibrated = np.load(f"files/uncalibrated datavolt/uncalibrated_datavolt_{run}_{board}.npy")
    calibration_info = np.load(f"files/calibration info/calibration_info_{board}.npy")

    event_size = len(uncalibrated[0])
    datavolt = np.full((16, event_size, 4094), -1, dtype=np.uint16)
    for ch in range(16):
        for event in range(event_size):
            datavolt[ch, event, :] = revert_single_event_to_uncalibrated(uncalibrated[ch, event, :],
                                                                         header[ch, event],
                                                                         -calibration_info[ch])
    np.save(f"files/uncalibrated datavolt/calibrated_datavolt_{run}_{board}.npy", datavolt)
    print(f"calibrated datavolt {run} {board} file "
          f"saved as files/uncalibrated datavolt/calibrated_datavolt_{run}_{board}.npy")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                                Noise detection

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# 단일 datavolt를 cell값 기준으로 바꿔서 내보내기. 다만 중간에 잘린 cell값은 4095로 채워짐 (길이 4094 -> 4096)
def single_datavolt_cell_align(single_event, last_tdc_cell_n):

    extended_datavolt = np.append(single_event, [4095, 4095])
    shifted_datavolt = np.roll(extended_datavolt, -4093 + last_tdc_cell_n)

    return shifted_datavolt


# 이상하게 나타나는 노이즈의 위상이 cell위치에 따라 바뀌지 않음을 확인. 여러 개의 run을 사용
def plot_noisy_event(run_list, board):
    header_dict = {run: np.load(f"files/header/header_{run}_{board}.npy") for run in run_list}
    datavolt_dict = {run: np.load(f"files/datavolt/datavolt_{run}_{board}.npy") for run in run_list}
    for run in run_list:
        event_size = len(datavolt_dict[run][0])
        for ch in range(16):
            for event in range(event_size):
                if np.sum(datavolt_dict[run][ch, event, :] <= 3500) >= 4050:
                    shift = header_dict[run][ch, event]
                    extended_datavolt = np.append(datavolt_dict[run][ch, event, :], [4095, 4095])
                    shifted_datavolt = np.roll(extended_datavolt, -4093 + shift)
                    plot_single_array(shifted_datavolt,
                                      f"Board.{board} Run.{run} Ch.{ch} Event.{event} shifted datavolt",
                                      "TDC", "ADC", [0, 3800], None, "line", "ADC",
                                      [1, 1025, 2049, 3073], [],
                                      0, 1, "files/images/shifted datavolt/")
        print(f"Board.{board} Run.{run} shifted datavolt file "
              f"saved as files/images/shifted datavolt/Board.{board} Run.{run} Ch.xx Event. xx shifted datavolt.jpg")
    return


# 각 채널의 노이즈가 없는 only dark current event들의 평균과 표준편차를 계산 이는 threshold를 설정하기 위한 용도
def datavolt_mean_std(run, board):
    datavolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    mean_std_list = []
    event_size = len(datavolt[0])
    for ch in range(16):
        std_less_than_2_datavolt_list = []
        count = 0
        for event in range(event_size):
            if count == 100:
                break
            if np.std(datavolt[ch, event, :4060]) < 2:
                std_less_than_2_datavolt_list.append(datavolt[ch, event, :4060])
                count += 1

        std_less_than_2_datavolt_array = np.array(std_less_than_2_datavolt_list)
        flatten_list = std_less_than_2_datavolt_array.flatten()

        mean_std_list.append([np.mean(flatten_list), np.std(flatten_list)])
    print(f"Run.{run} mean, std calculated")
    return mean_std_list


# datavolt, header를 이용해서 샘플 노이즈를 추출 및 저장. 여러 개의 run을 사용. np.int16
def extract_noise_sample(run_list, board):
    header_dict = {run: np.load(f"files/header/header_{run}_{board}.npy") for run in run_list}
    datavolt_dict = {run: np.load(f"files/datavolt/datavolt_{run}_{board}.npy") for run in run_list}

    noise_sample = np.zeros((16, 4096), dtype=np.int16)
    division = 8
    length = 4096 // division
    total_regions = 16 * division
    terminate = False

    for run in run_list:
        if terminate:
            break

        event_size = len(datavolt_dict[run][0])
        mean_std = datavolt_mean_std(run, board)

        for ch in range(16):
            if terminate:
                break

            ch_mean = mean_std[0][0]
            ch_std = mean_std[0][1]
            threshold = ch_mean - 10 * ch_std

            for event in range(event_size):

                shift = header_dict[run][ch, event]
                shifted_datavolt = np.append(datavolt_dict[run][ch, event, :], [4095, 4095])
                shifted_datavolt = np.roll(shifted_datavolt, shift - 4096)

                for region in range(division):

                    datavolt_region = shifted_datavolt[length * region: length * (region + 1)]

                    if (min(datavolt_region) > 2700 and np.std(datavolt_region) > 50
                            and np.mean(datavolt_region) > 3100 and np.max(datavolt_region) < 4095):

                        count = 0

                        for tdc in range(length):
                            if datavolt_region[tdc] >= threshold:
                                count += 1

                        if count <= 10 and noise_sample[ch, length * region] == 0:
                            noise_sample[ch, length * region:length * (region + 1)] = datavolt_region

            remain_regions = (np.size(noise_sample) - np.count_nonzero(noise_sample)) // length
            print(f"Remaining regions: {remain_regions}/{total_regions}")
            if remain_regions == 0:
                print("Early stop: All regions collected")
                terminate = True
                break

    # for ch in range(16):
    #     plot_single_array(noise sample[ch], f"Ch.{ch} noise sample", "TDC", "ADC",
    #                       ylim=[0, 3800], mode="line", datatype="ADC", points=None, show=1,
    #                       save=0, path=None)

    noise_sample = noise_sample.astype(np.int16)
    np.save(f"files/noise sample/noise_sample_{board}.npy", noise_sample)
    print(f"noise sample {board} file saved as files/noise sample/noise_sample_{board}.npy")


# datavolt와 nosie_sample로 점수 계산
@njit
def find_ratio(datavolt, noise_sample):
    max_data_size = 16 * len(datavolt[0])  # 최대 크기 예측
    data_array = np.zeros((max_data_size, 6), dtype=np.int32)  # 미리 공간 할당
    data_index = 0  # 현재 저장할 위치

    for ch in range(16):
        noise_smp = noise_sample[ch]
        trend1 = np.zeros(4096, dtype=np.int8)
        trend1[0] = 1 if noise_smp[0] > noise_smp[-1] else -1 if noise_smp[0] < noise_smp[-1] else 0
        trend1[1:] = np.sign(noise_smp[1:] - noise_smp[:-1])

        for evt in range(len(datavolt[0])):
            if np.std(datavolt[ch, evt, :4060]) >= 2:
                single_event = datavolt[ch, evt, :]

                trend2 = np.zeros(4094, dtype=np.int8)
                trend2[0] = 1 if single_event[0] > single_event[-1] else -1 if single_event[0] < single_event[-1] else 0
                trend2[1:] = np.sign(single_event[1:] - single_event[:-1])

                score_array = np.zeros(4096, dtype=np.int32)
                max_score = 0
                second_max_score = 0
                best_shift = 0

                for shift in range(4096):
                    shifted_noise_trend = np.concatenate((trend1[-shift:], trend1[:-shift]))
                    score = 0
                    streak = 0

                    for t1, t2 in zip(shifted_noise_trend, trend2):
                        if t1 == t2:
                            streak += 1
                            score += streak
                        else:
                            streak = 0

                    score_array[shift] = score
                    if score > max_score:
                        second_max_score = max_score
                        max_score = score
                        best_shift = shift
                    elif score > second_max_score:
                        second_max_score = score

                # 데이터 저장
                data_array[data_index] = [ch, evt, best_shift, max_score, second_max_score, int(np.mean(score_array))]
                data_index += 1
        print(f"Ch.{ch} score calcualted.")
    return data_array[:data_index]  # 사용된 데이터만 반환


# find_ratio 방식 사용의 검증을 위한 시뮬레이션 (그림 그리는 기능 자체 포함)
def plot_find_ratio_simulation(iteration_number=100,
                               min_length=10, max_length=300):

    print(f"Total iteration:{16 * iteration_number * (max_length - min_length + 1)}")

    # 고정시드 0 ~ 50 사이의 랜덤 정수 값을 가지는 길이 4096의 array (noise sample)
    noise_sample = np.zeros((16, 4096), dtype=np.int16)
    for ch in range(16):
        noise_sample[ch] = np.random.randint(0, 51, 4096, dtype=np.int16)

    shift_and_score_ratio = []

    # 시뮬레이션 시행
    for length in range(min_length, max_length + 1):

        datavolt = np.zeros((16, iteration_number, 4094), dtype=np.uint16)
        start_array = np.zeros((16, iteration_number), dtype=np.int16)

        for ch in range(16):
            for event in range(iteration_number):
                # 매번 변하는 0 ~ 50 사이의 랜덤 정수 값을 가지는 길이 4094의 array (datavolt)
                datavolt[ch, event] = np.random.randint(0, 51, 4094, dtype=np.int16)

                # noise_sample에서 해당 시작점, 길이만큼의 구간 추출.
                selected_noise_sample_region = noise_sample[ch][4096 - length:]

                start = np.random.randint(0, 4094 - length + 1)
                start_array[ch, event] = start
                # datavolt의 start2 위치시작점에서 해당 노이즈 구간을 삽입 datavolt의 길이가 4094이므로 순환구조 형식으로 채워야 함
                datavolt[ch, event, start: start + length] = selected_noise_sample_region

        print(f"{length} TDC noise embeded datavolt, noise sample generated")

        score = find_ratio(datavolt, noise_sample)

        for ch in range(16):
            for event in range(iteration_number):
                s = ch * iteration_number + event
                score_ratio = (score[s][3] - score[s][5]) / abs(score[s][4] - score[s][5])
                shift_and_score_ratio.append([length,
                                              score_ratio,
                                              score[s][2] - start_array[ch, event] - length])

    print("All iteration processed. Start drawing plot")

    # 플롯 그리기
    plt.figure(figsize=(18, 6))

    colors = ['blue' if a == 0 else 'red' for _, _, a in shift_and_score_ratio]
    xs = [x for x, _, _ in shift_and_score_ratio]
    ys = [np.sqrt(y) for _, y, _ in shift_and_score_ratio]

    plt.scatter(xs, ys, c=colors, alpha=1/iteration_number)

    threshold = np.sqrt(2)

    # 그 후 y축에서 x축과 평행하게 얇은 빨간 점선을 표시 후 그 threshold 라벨 표시
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold: {threshold}')
    plt.legend()

    # 그래프 꾸미기
    plt.xlabel('Embeded noise region length (TDC)')
    plt.xlim(0, max_length + 10)
    plt.ylabel('Score ratio ^ (1/2)')
    # plt.yscale('log')
    plt.title('find_ratio simulation result')
    plt.grid(True)

    plt.savefig(f"files/images/find ratio simulation/find_ratio_simulation_with_"
                f"{16 * iteration_number * (max_length - min_length + 1)}_iteration.png", dpi=300)

    print(f"find ratio simulation with {16 * iteration_number * (max_length - min_length + 1)} iteration saved as"
          f"files/images/find ratio simulation/find_ratio_simulation_with_"
          f"{16 * iteration_number * (max_length - min_length + 1)}_iteration.png")

    plt.show()
    plt.close()

    return


# negative 펄스로 구분되는 구간(ADC > 2500) & 첫 구간의 끝 위치를 계산해서 내보냄
def find_regions_over_threshold(array, threshold=2500, min_length=50):
    binary_array = np.ones(4094, dtype=np.int8)

    indices = np.where(array <= threshold)[0]
    split_points = np.where(np.diff(indices) > 1)[0]
    segments = np.split(indices, split_points + 1)

    for segment in segments:
        if segment.size > 0:  # 구간이 비어있지 않은 경우만 처리
            if array[segment].min() <= 300:
                binary_array[segment] = 0

    # 조건 1 모든 값이 2500 이상
    if np.sum(binary_array) == 4094:
        return np.array([]).astype(np.int16), 0

    # 조건 2 구간이 반드시 나눠짐
    regions = []
    start = None
    first_region_found = False

    for i, val in enumerate(binary_array):
        if val == 1 and start is None:
            start = i  # Start of a new region
        elif val == 0 and start is not None:
            # End of a region
            if not first_region_found:
                # 첫 번째 구간은 길이 제한 없음
                regions.append([start, i])
                first_region_found = True
            else:
                # 나머지 구간은 길이 제한 적용
                if i - start > min_length:
                    regions.append([start, i])
            start = None
    if start is not None:
        if not first_region_found:
            # 마지막 구간이 첫 번째 구간인 경우 길이 제한 없음
            regions.append([start, len(binary_array)])
        elif len(binary_array) - start > min_length:
            regions.append([start, len(binary_array)])

    first_region_end_pos = regions[0][1] if regions else 0

    return np.array(regions).astype(np.int16), first_region_end_pos


# array에서 주어진 범위 이내의 ADC 값이 연속되는 구간을 파악
def regions_bounded_test(array, indice,
                         upper_bound=3100, below_bound=2600):
    condition = (array < upper_bound) & (array > below_bound)
    con = np.where(condition)[0]

    if len(con) > 0:
        starts = []
        ends = []
        current_start = con[0]

        for i in range(1, len(con)):
            if con[i] != con[i - 1] + 1:  # 연속이 끊어지면
                starts.append(current_start)
                ends.append(con[i - 1])
                current_start = con[i]
        starts.append(current_start)
        ends.append(con[-1])

        # 5. 길이가 50 이상인 연속 구간만 추출
        long_enough_ranges = [(s, e) for s, e in zip(starts, ends) if e - s + 1 >= 50]
        if long_enough_ranges:
            merged_starts = [long_enough_ranges[0][0]]
            merged_ends = [long_enough_ranges[0][1]]

            for i in range(1, len(long_enough_ranges)):
                if long_enough_ranges[i][0] - merged_ends[-1] <= 10:  # 간격이 10 이하이면 병합
                    merged_ends[-1] = long_enough_ranges[i][1]
                else:
                    merged_starts.append(long_enough_ranges[i][0])
                    merged_ends.append(long_enough_ranges[i][1])

            f_start = merged_starts[0]
            if f_start < 100:
                f_start = 0

            f_end = merged_ends[-1]
            if indice != 0 and f_end < indice and indice - f_end < 100:
                f_end = indice - 1
            if f_end > 4093 - 100:
                f_end = 4093

            return [f_start, f_end], 1

    return [], 0


# 단일 datavolt, noise sample, header를 이용해서 trend가 같은 위치를 파악
def datavolt_similiarity_indice(list_a, list_b, shift_v):

    # 오버플로우 방지용 16bit로 되어있는 것을 그대로 이용했더니 np.sign할 때 오버플로우 남.
    list_a = list_a.astype(np.int32)

    # trend1 - single datavolt event
    trend1 = np.zeros(4094, dtype=np.int8)
    trend1[0] = 1 if list_a[0] > list_a[-1] else -1 if list_a[0] < list_a[-1] else 0
    trend1[1:] = np.sign(list_a[1:] - list_a[:-1])

    # trend2 - noise sample
    trend2 = np.zeros(4096, dtype=np.int8)
    trend2[0] = 1 if list_b[0] > list_b[-1] else -1 if list_b[0] < list_b[-1] else 0
    trend2[1:] = np.sign(list_b[1:] - list_b[:-1])

    shifted_noise_trend = np.concatenate((trend2[-shift_v:], trend2[:-shift_v]))

    score = 0
    streak = 0
    for t1, t2 in zip(shifted_noise_trend, trend1):
        if t1 == t2:
            streak += 1
            score += streak
        else:
            streak = 0

    matching_indices = [1 if trend1[i] == shifted_noise_trend[i] else 0 for i in range(len(trend1))]
    return matching_indices


# 단일 datavolt와 noise_sample간의 pearson 상관관계를 계산
@njit
def datavolt_noise_sample_pearson_correlation(list_a, list_b,
                                              ws_min=5, ws_max=100):

    # 각 위치의 점수의 최댓값 max_vote_v는 기본 4940이다. (5 + 6 + ... + 100)
    max_vote_v = (ws_max - ws_min + 1) * (ws_min + ws_max) // 2
    votes = np.zeros(4096)

    for ws in range(ws_min, ws_max + 1):
        for s in range(4096):
            # Use rolling window with pre-computed indices
            idx1 = np.arange(s, s + ws) % 4096
            idx2 = np.arange(s, s + ws) % 4096

            x = list_a[idx1]
            y = list_b[idx2]

            mean_x = np.sum(x) / ws
            mean_y = np.sum(y) / ws

            # Precompute covariance and standard deviation
            cov = np.sum((x - mean_x) * (y - mean_y))

            std_x = np.sqrt(np.sum(x ** 2) - mean_x ** 2 * ws)
            std_y = np.sqrt(np.sum(y ** 2) - mean_y ** 2 * ws)

            # Avoid division by zero
            if std_x == 0 or std_y == 0:
                correlation = 0.0

            else:
                correlation = cov / (std_x * std_y)

            # Update votes if correlation threshold is met
            if correlation > 0.7:
                votes[idx1] += 1

    sliced_votes = votes[:4094]

    return sliced_votes, max_vote_v


# datavolt, noise_sample을 이용해서 노이즈 구간 파악 및 결과 저장 np.int32
def noise_region_detect(run, board):

    print(f"Borad.{board} Run.{run} noise region detection started")

    datavolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    noise_sample = np.load(f"files/noise sample/noise_sample_{board}.npy")
    header = np.load(f"files/header/header_{run}_{board}.npy")

    # find_ratio 실행 후 점수 정보 계산
    scores = find_ratio(datavolt, noise_sample)
    print(f"Total test events: {len(scores)}")

    # 각 채널별 dark current 이벤트 100개의 평균값
    mean_std = datavolt_mean_std(run, board)

    # 노이즈 판단 정보를 담을 리스트
    noise_detection_result = np.zeros((len(scores), 6), dtype=np.int32)

    for index, score in enumerate(scores):

        score_ratio = (score[3] - score[5]) / abs(score[4] - score[5])
        ch = int(score[0])
        event = score[1]
        s_datavolt = datavolt[ch, event, :]
        real_shift = 4096 - header[ch, event]
        shift = score[2]
        mean = mean_std[ch][0]

        # 2500 이상인 구간들, 첫 구간의 맨 끝 인덱스
        over_regions, region1_end = find_regions_over_threshold(s_datavolt)

        # 유사도 극악 노이즈 구간들 및 존재 유무
        bounded_regions, exists = regions_bounded_test(s_datavolt, region1_end)

        # 실제 맞는 shift값을 이용, datavolt와 noise_sample의 trend가 맞는 부분=1, 아닌 부분은 0으로 해서 추출
        matching_indice = datavolt_similiarity_indice(s_datavolt, noise_sample[ch], real_shift)

        noise_start = 0
        noise_end = 4093
        types = 0

        # 노이즈가 확실히 있음
        if score_ratio >= 2 and shift == real_shift:

            # 노이즈가 확실히 있음 / negative 펄스가 있음 => shift 값 이전, 이후에는 유사도가 엄청 낮다는 특성을 이용
            if region1_end != 0:

                types = 1

                # shift 값보다 한칸 왼쪽으로 shift를 할 때, trend 파악
                matching_indice_prev = datavolt_similiarity_indice(s_datavolt, noise_sample[ch], real_shift - 1)
                # shift 값보다 한칸 오른쪽으로 shift를 할 때, trend 파악
                matching_indice_post = datavolt_similiarity_indice(s_datavolt, noise_sample[ch], real_shift + 1)

                # 왼쪽 한칸 밀림, 기존, 오른쪽 한칸 밀림 이 3개를 묶기
                m_indices_list = [matching_indice_prev, matching_indice, matching_indice_post]

                phase_shift_ratios = np.zeros((3, len(over_regions)))

                for i in range(3):
                    for j, region in enumerate(over_regions):
                        r_s = region[0]
                        r_e = region[1]
                        phase_shift_ratios[i, j] = np.sum(m_indices_list[i][r_s: r_e]) / (r_e - r_s)

                edge_list = []
                for j, region in enumerate(over_regions):
                    if phase_shift_ratios[1, j] - ((phase_shift_ratios[0, j] + phase_shift_ratios[2, j]) / 2) > 0.1:
                        edge_list.append(region[0])
                        edge_list.append(region[1])

                if edge_list:
                    noise_start = np.min(edge_list)
                    noise_end = min(np.max(edge_list), 4093)

            # 노이즈가 확실히 있음 / negative 펄스가 없음 => 가변 슬라이싱 윈도우 기반, 피어슨 상관 계수로 노이즈 구간으로 판단
            else:

                types = 2

                extended_list_a = np.append(s_datavolt, [int(mean), int(mean)])
                shifted_list_b = np.concatenate((noise_sample[ch][-real_shift:], noise_sample[ch][:-real_shift]))
                extended_list_a = extended_list_a.astype(np.int32)
                shifted_list_b = shifted_list_b.astype(np.int32)

                pearson_vote, max_vote = datavolt_noise_sample_pearson_correlation(extended_list_a, shifted_list_b)

                """
                print(ch, event)
                plot_single_array(pearson_vote, 1, 1, 1,
                                  ylim=[0, max_vote + 10], mode="line", datatype="else", points=None,
                                  show=1, save=0, path=None)
                plot_single_array(s_datavolt, 1, 1, 1,
                                  ylim=[0, 3800], mode="line", datatype="else", points=None,
                                  show=1, save=0, path=None)
                """

                # pearson_vote로 노이즈가 있는 구간 없애기 이건 한번 설정을 100으로 해보기
                indices = np.where(pearson_vote >= 100)[0]
                diff = np.diff(indices)
                is_continuous = np.concatenate(([False], diff == 1)) | np.concatenate((diff == 1, [False]))
                filtered_indices = indices[is_continuous]
                if len(filtered_indices) != 0:
                    noise_start = indices[0]
                    noise_end = indices[-1]

                    if noise_start < 100:
                        noise_start = 0
                    if noise_end > 4093 - 100:
                        noise_end = 4093

        # 노이즈의 유무가 확실치 않음
        else:

            noise_start = 0
            noise_end = 0

            # print(ch, event, region1_end)

            # 노이즈가 확실치 않음 / negative 펄스가 있음 => 첫 구간(>2500) 길이 찗은 구간 노이즈 유뮤 판단, region1_end 이용
            if region1_end != 0 and region1_end <= 100:

                if region1_end < 10:
                    types = 3
                    noise_start = 0
                    noise_end = region1_end
                elif np.mean(s_datavolt[:10]) < 3600:
                    types = 3
                    noise_start = 0
                    noise_end = region1_end

            # 노이즈가 확실치 않음 / negative 펄스가 없음 => matching indice 이용
            elif region1_end == 0:

                # matching indice에서 슬라이싱 윈도우를 100으로 잡고 비율 0.7 이상 길이 100이상인 구간 자르기
                ratio_list = np.zeros(4094 - 100 + 1)
                candidate_list = []
                for sw in range(4094 - 100 + 1):
                    ratio_list[sw] = np.sum(matching_indice[sw:sw + 100]) / 100
                    if np.sum(matching_indice[sw:sw + 100]) / 100 > 0.7:
                        candidate_list.append(sw)
                        candidate_list.append(sw + 99)

                if candidate_list:

                    types = 4

                    # print(ch, event)

                    noise_start = candidate_list[0]
                    noise_end = candidate_list[-1]

                    # 앞이나 뒤에 노이즈랑 붙은 100이하의 구간이 있을 수 있음. 이는 확장하여 제거(노이즈나 다름 없음)
                    if noise_start < 100:
                        noise_start = 0
                    if noise_end > 4093 - 100:
                        noise_end = 4093

        # 유사도가 극악으로 낮지만 확연히 노이즈인 구간도 같이 묶기
        if exists:
            noise_start = min(noise_start, bounded_regions[0])
            noise_end = max(noise_end, bounded_regions[1])
            types += 10

        # 최종적으로 잘라내기
        if noise_start < 10:
            noise_start = 0
        if noise_end > 4093 - 10:
            noise_end = 4093

        noise_detection_result[index] = [ch, event, real_shift, noise_start, noise_end, types]
        print(ch, event, "Start", noise_start, "End", noise_end, "Type", types)

        """
        plot_single_array(s_datavolt, 1, 1, 1,
                          ylim=[0, 3800], mode="line", datatype="else", points=None, span=[noise_start, noise_end + 1],
                          show=1, save=0, path=None)
        """

    # breakpoint()

    noise_detection_result = np.array(noise_detection_result)
    np.save(f"files/noise detection result/noise_detection_result_{run}_{board}.npy", noise_detection_result)

    print(f"Board.{board} Run.{run} noise detection result saved as "
          f"files/noise detection result/noise detection result/noise_detection_result_{run}_{board}.npy")

    return


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                                Detected Noise analysis

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# 노이즈가 있다고 판단한 이벤트들을 calibrated 되기 이전 상태로 되돌려서 보기
def plot_noise_detection_result(run, board,
                                calibration_off=0, noise_only=0,
                                show=0, save=0):

    datavolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    noise_detection_result = np.load(f"files/noise detection result/noise_detection_result_{run}_{board}.npy")
    header = np.load(f"files/header/header_{run}_{board}.npy")
    calibration_info = np.load(f"files/calibration info/calibration_info_{board}.npy")

    print(f"Total results on Board.{board} Run.{run}: {len(noise_detection_result)}")

    for index, result in enumerate(noise_detection_result):
        ch = result[0]
        event = result[1]
        noise_start = result[3]
        noise_end = result[4]
        types = result[5]

        if noise_only != 1 or types != 0:
            s_datavolt = datavolt[ch, event, :]
            s_header = header[ch, event]

            title = f"Board.{board} Run.{run} Ch.{ch} Event.{event} Type.{types} calibrated noise event"
            if calibration_off == 1:
                ch_calibration_info = calibration_info[ch]
                s_datavolt = revert_single_event_to_uncalibrated(s_datavolt, s_header, ch_calibration_info)
                title = f"Board.{board} Run.{run} Ch.{ch} Event.{event} Type.{types} uncalibrated noise event"

            plot_single_array(s_datavolt, title, "TDC", "ADC",
                              ylim=[-100, 3800], mode="line", datatype="else", points=None,
                              span=[noise_start, noise_end],
                              show=show, save=save, path="files/images/noise detection result/")

            if save == 1:
                print(f"({index}/{len(noise_detection_result)}) saved.")
        else:
            if save == 1:
                print(f"({index}/{len(noise_detection_result)}) skipped.")

    if calibration_off == 0:
        print(f"Board.{board} Run.{run} calibrated noise detection result saved as"
              f"files/images/noise detection result/"
              f"Board.{board} Run.{run} Ch.xx Event.xx Type.xx calibrated noise event.jpg")
    else:
        print(f"Board.{board} Run.{run} uncalibrated noise detection result saved as"
              f"files/images/noise detection result/"
              f"Board.{board} Run.{run} Ch.xx Event.xx Type.xx uncalibrated noise event.jpg")

    return


# 노이즈가 있다고 판단한 이벤트들을 같은 이벤트 넘버를 가진 다른 16채널을 동시에 그리기
def plot_noise_included_channels_set(run, board):

    import matplotlib
    matplotlib.use('Agg')  # Agg 백엔드 설정

    datavolt = np.load(f"files/datavolt/datavolt_{run}_{board}.npy")
    noise = np.load(f"files/noise detection result/noise_detection_result_{run}_{board}.npy")
    mean_std = datavolt_mean_std(run, board)

    # 노이즈가 있다고 판단한 이벤트
    noise_inc_events = [subarray[1] for subarray in noise if subarray[5] != 0]

    noise_inc_single_event = []
    for item in noise_inc_events:
        if item not in noise_inc_single_event:
            noise_inc_single_event.append(item)
    noise_inc_single_event.sort()

    print(f"Total number of channel sets containing at least one noisy event:{len(noise_inc_single_event)}")

    for event in noise_inc_single_event:

        # 서브플롯 생성
        fig, axes = plt.subplots(16, 1, figsize=(12, 12), sharex='all')  # X축 공유, figsize 조정

        # 각 채널의 데이터를 서브플롯에 그리기
        for ch in range(16):

            noise_start, noise_end = 0, 0  # 기본값

            for noise_item in noise:
                if noise_item[0] == ch and noise_item[1] == event:
                    noise_start = int(noise_item[3])
                    noise_end = int(noise_item[4])
                    break

            # 기본 데이터 그리기
            axes[ch].plot(datavolt[ch, event, :], label="Signal")

            # noise_start, noise_end가 0이 아니면 빨간색으로 표시
            if noise_start != 0 or noise_end != 0:
                axes[ch].plot(
                    range(noise_start, noise_end + 1),
                    datavolt[ch, event, noise_start:noise_end + 1],
                    color="red")

            axes[ch].set_ylabel(f"Ch {ch}", rotation=0, labelpad=15)  # Y축 라벨
            axes[ch].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)  # X축 제거

            ymax = mean_std[ch][0] + 30 * mean_std[ch][1]
            ymin = ymax - 1000
            axes[ch].set_ylim(ymin, ymax)

            if ch == 15:  # 마지막 서브플롯만 X축 라벨 표시
                axes[ch].tick_params(axis="x", which="both", bottom=True, labelbottom=True)
                axes[ch].set_xlabel("Index")

        # 제목 제거 및 그래프 간격 조정
        plt.tight_layout(h_pad=0.5)  # 서브플롯 간격 최소화

        plt.savefig(f"files/images/noise included channels set"
                    f"/Board.{board}_Run.{run}_Event.{event}_noise_included_channels.jpg")
        plt.close()

    print(f"Board.{board} Run.{run} simultaneous_noise saved as "
          f"files/images/noise included channels set/Board.xx_Run.xx_Event.xx_noise_included_channels_set.jpg")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                                                Plot

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


# 1차원 array를 그래프로 나타내기
def plot_single_array(array, title="", xaxis="TDC", yaxis="ADC",
                      ylim=None, xlim=None, mode="line", datatype="else", points=None, span=None,
                      show=0, save=0, path=None):
    """
    1차원 배열을 그래프로 그리기

    Parameters:
    - array: 1차원 데이터 배열

    # 그림 제목, 축 설정
    - title: 그래프 제목
    - xaxis: x축 이름
    - yaxis: y축 이름
    - ylim: y축 범위, [min, max] 형식. 비어 있으면 값의 최소-최대 범위를 사용

    # 그림 그리는 모드 및 부가 기능
    - mode: "line" 또는 "dot", 각각 선 그래프와 점 그래프를 나타냄
    - datatype: "ADC" 또는 "else" 여기서 ADC일 경우, 값이 4095인 부분은 공백으로 처리
    - points: [] 형식으로 그 위치의 값을 빨간색 점으로 표시
    - span: [] 형식으로 시작x <= x <= 끝점x 포함한 부분의 영역의 데이터 선 색을 빨간색으로 표시

    # 그림 표시 및 저장
    - show: 그림 표시
    - save: 저장
    - path: 저장 위치
    """

    # 저장 모드일 때는 show가 동작하지 않음
    if save == 1:
        import matplotlib
        matplotlib.use('Agg')  # Agg 백엔드 설정

    if show == save == 1:
        raise ValueError("Can't show image while save mode is oon. Turn off 1 mode between show / save.")
    elif show == save == 0:
        raise ValueError("Both show / save mode offed. What are you going to do?")

    # "ADC"인 경우 4095 값을 NaN으로 처리
    if datatype == "ADC":
        array = np.where(array == 4095, np.nan, array)

    plt.figure(figsize=(12, 6))

    if mode == "line":
        plt.plot(array, drawstyle='steps-post', label='Data')  # 계단형 선 그래프
    elif mode == "dot":
        plt.plot(array, 'o', label='Data')  # 점 그래프

    # 특정 위치에 빨간 점 추가
    if points is not None:
        for point in points:
            if 0 <= point < len(array) and not np.isnan(array[point]):  # NaN이 아닌 유효한 인덱스인지 확인
                plt.scatter(point, array[point], color='red', s=30)  # 빨간 점으로 표시

    # span 영역 강조
    if span is not None:
        start = int(span[0])
        end = int(span[1])

        if start == end == 0:
            pass  # 아무 작업도 하지 않음
        else:
            if mode == "line":
                plt.plot(range(start, end + 1), array[start:end + 1], drawstyle='steps-post', color='red')
            else:
                plt.plot(range(start, end + 1), array[start:end + 1], 'o', color='red')
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)

    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim(np.nanmin(array), np.nanmax(array))  # NaN 제외한 최소/최대값 사용

    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim(0, 4095)  # NaN 제외한 최소/최대값 사용

    plt.grid(True)
    if save == 1:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f"{path}{title}.jpg", dpi=100)
    if show == 1:
        plt.show()
    plt.close()


# file_path에 있는 모든 이미지를 length (초) 시간 만큼 하나의 gif로 이름을 name으로 해서 생성
def gif_generator(files_path, length, name):
    """
    주어진 파일 경로의 모든 이미지를 하나의 GIF로 생성하며, 진행률 표시를 추가합니다.

    Parameters:
    - files_path (str): 이미지가 저장된 디렉토리 경로
    - length (int): GIF의 전체 길이 (초)
    - name (str): 생성할 GIF의 이름
    """
    # 이미지 파일 로드
    images = []
    valid_extensions = ('.png', '.jpg')  # 유효한 이미지 확장자
    for file_name in sorted(os.listdir(files_path)):  # 정렬된 파일 리스트
        if file_name.lower().endswith(valid_extensions):  # 확장자 확인
            images.append(Image.open(os.path.join(files_path, file_name)))

    if not images:
        print("No valid images found in the specified path.")
        return None

    # 프레임당 시간 계산
    duration = int((length / len(images)) * 1000)  # 각 프레임 시간 (ms)

    # 진행률 표시 추가
    processed_images = []
    font = ImageFont.load_default()  # 기본 폰트 사용
    for i, img in enumerate(images):
        # 새로운 이미지 생성 (캔버스 크기 확장)
        width, height = img.size
        new_height = height + 30  # progress bar 공간 추가
        new_image = Image.new("RGB", (width, new_height), "white")
        new_image.paste(img, (0, 0))

        # 진행률 계산
        progress = int((i + 1) / len(images) * 100)

        # 진행률 바 그리기
        draw = ImageDraw.Draw(new_image)
        progress_bar_width = int((i + 1) / len(images) * width)
        draw.rectangle([0, height, progress_bar_width, new_height], fill="blue")

        # 진행률 텍스트 추가
        text = f"{progress}%"
        text_bbox = draw.textbbox((0, 0), text, font=font)  # 텍스트 크기 계산
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((width - text_width) // 2, height + (30 - text_height) // 2)
        draw.text(text_position, text, fill="black", font=font)

        # 가공된 이미지 추가
        processed_images.append(new_image)

    # GIF 저장 경로
    gif_path = os.path.join(files_path, f"{name}.gif")

    # GIF 생성 및 저장
    print("Starting GIF creation...")
    processed_images[0].save(
        gif_path,
        save_all=True,
        append_images=processed_images[1:],  # 한 번에 모든 프레임 추가
        duration=duration,
        loop=0,
    )

    print(f"{name} saved as {gif_path}")
