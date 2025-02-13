import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib
matplotlib.font_manager._rebuild()

# 한글 폰트 설정 (시스템에 설치된 한글 폰트 경로를 사용하세요)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 폰트 경로 확인
font_prop = font_manager.FontProperties(fname=font_path)
font_name = font_prop.get_name()
rc('font', family=font_name)

def sort_data(brightness, success):
    zipped = zip(brightness, success)
    sorted_pairs = sorted(zipped)
    brightness_sorted, success_sorted = zip(*sorted_pairs)
    return list(brightness_sorted), list(success_sorted)

# 데이터 입력
# 모델 1
brightness1 = [1.0, 0.5, 0.4]
success1 = [96.40, 98.40, 12.20]

# 모델 2
brightness2 = [1.0, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
success2 = [25.40, 14.20, 21.60, 38.60, 65.00, 96.00, 94.80, 94.60, 94.60, 16.40]
brightness2, success2 = sort_data(brightness2, success2)

# 모델 3
brightness3 = [1.0, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.045]
success3 = [82.40, 80.80, 85.20, 83.40, 84.80, 83.80, 81.70, 86.00, 79.80, 82.60, 82.80]
brightness3, success3 = sort_data(brightness3, success3)

# 모델 4
brightness4 = [1.0, 0.5, 0.4, 0.35, 0.3, 0.25]
success4 = [98.40, 98.40, 99.00, 94.20, 28.80, 13.00]
brightness4, success4 = sort_data(brightness4, success4)

# 모델 5
brightness5 = [1.0, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.045]
success5 = [51.80, 72.20, 80.40, 77.20, 85.40, 87.00, 87.00, 92.60, 96.60, 76.20, 47.00]
brightness5, success5 = sort_data(brightness5, success5)

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.xlim(1.0, 0.0)
plt.ylim(0.0, 100.0)

plt.plot(brightness1, success1, label='① VAE_Encoder 와 Actor-Critic Network를\nDRL training(1.0)', color='orange')
plt.plot(brightness2, success2, label='② VAE_Encoder 와 Actor-Critic Network를\nDRL training(0.2)', color='purple')
plt.plot(brightness3, success3, label='③ Pretrained-VAE\n(1.0, 0.1 -> 1.0 / weight frozen)\n+ DRL training(1.0)', color='green')
plt.plot(brightness4, success4, label='④ Pretrained-VAE\n(1.0, 0.1 ->1.0 / weight updated)\n+ DRL training(1.0)', color='red', linewidth=5)
plt.plot(brightness5, success5, label='⑤ Pretrained-VAE\n(0.2 -> 1.0 / weight updated)\n+ DRL training(0.2)', color='blue', linewidth=5)

plt.xlabel('조도 (Brightness)')
plt.ylabel('성공률 (Success Rate)')
plt.title('실험 결과_방법 비교 그래프')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()