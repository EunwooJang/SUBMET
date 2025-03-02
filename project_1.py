from main import *
import shutil
import tkinter as tk
from tkinter import filedialog

# 설정된 변수
run = 103
board = 10
test = 10

# GUI로 파일 선택
root = tk.Tk()
root.withdraw()  # GUI 창을 숨김
selected_file = filedialog.askopenfilename(title="Select a file")
root.quit()
root.destroy()

# 대상 디렉토리 및 새 파일 이름 설정
target_directory = "files/raw"
new_file_name = f"raw_{run}_{board}.dat"
target_path = os.path.join(target_directory, new_file_name)

shutil.move(selected_file, target_path)
print(f"File moved and renamed to: {target_path}")

# datavolt 및 헤더정보  추출 및 저장
save_raw_to_datavolt(run, board)
save_raw_to_header(run, board)

# 히스토그램 그리기
datavolt = load_datavolt(run, board)
valid_ch = [2, 3]  # 이건 데이터 보고 변경해야 함
# header = load_header(run, board)

# Persistence Plot 생성

tdc_range = 0

for ch in valid_ch:  # 각 채널에 대해 반복
    # 2D 히스토그램 초기화 (x: index, y: value)
    persistence = np.zeros((4094, 4096), dtype=int)
    for event in range(len(datavolt[ch])):  # 각 이벤트에 대해 반복
        for idx in range(tdc_range, 4094):  # 각 인덱스에 대해 반복
            value = datavolt[ch, event, idx]
            persistence[idx, value] += 1  # (index, value) 빈도 누적
    # Persistence 플롯 그리기 (로그 변환 없음)
    plt.figure(figsize=(20, 8))
    plt.imshow(
        persistence.T,  # .T로 전치 (index: x축, value: y축)
        aspect='auto',
        cmap='inferno',
        origin='lower'
    )
    plt.colorbar(label="Frequency")
    plt.title(f"Persistence Plot for Channel {ch}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.ylim(3600, 3800)
    plt.xlim(tdc_range, 4094)
    plt.tight_layout()
    plt.show()
    plt.close()

thresholds = [2000, 2500, 3000, 3500]

for ch in valid_ch:
    for threshold in thresholds:
        max_crossing = 0
        min_crossing = 0
        for event in range(len(datavolt[0])):
            above_threshold = datavolt[ch, event, :] > threshold
            crossings = np.diff(above_threshold.astype(int))
            crossing_count = np.sum(np.abs(crossings))
            if crossing_count < min_crossing:
                min_crossing = crossing_count
            if crossing_count > max_crossing:
                max_crossing = crossing_count
        print(f"Ch.{ch} Threshold.{threshold}")
        print(f"{min_crossing} {max_crossing}")
