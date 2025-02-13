import numpy as np

# 파일 경로
file_path = 'images/converted_depth2/converted_npy/10000.npy'

# .npy 파일 로드
try:
    array = np.load(file_path)
    print(f"Array shape: {array.shape}")
except Exception as e:
    print(f"Error loading array from {file_path}: {e}")