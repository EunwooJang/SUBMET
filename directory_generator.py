from main import *


# 디렉토리 구조 정의 main.py와 files은 같은 위치에 있음
directory_structure = {
    "files": {
        "calibration info": None,
        "datavolt": None,
        "document": None,
        "header": None,
        "images": {
            "find ratio simulation": None,
            "noise detection result": None,
            "noise included channels set": None,
            "shifted datavolt": None
        },
        "noise detection result": None,
        "noise sample": None,
        "raw": None,
        "uncalibrated datavolt": None
    }
}

base_path = os.path.dirname(os.path.abspath(__file__))
directory_generator(base_path, directory_structure)
