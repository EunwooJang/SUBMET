from main import *


# 디렉토리 구조 정의
directory_structure = {
    "files": {
        "calibration info": None,
        "datavolt": None,
        "document": None,
        "header": None,
        "images": {
            "extra header": None,
            "find ratio simulation": None,
            "noise detection only noise with cell align": None,
            "noise detection result": None,
            "noise included channels set": None,
            "shifted datavolt": None,
            "noise std vs mean": None
        },
        "noise detection result": None,
        "noise sample": None,
        "raw": None,
        "uncalibrated datavolt": None
    }
}

base_path = os.path.dirname(os.path.abspath(__file__))
directory_generator(base_path, directory_structure)
