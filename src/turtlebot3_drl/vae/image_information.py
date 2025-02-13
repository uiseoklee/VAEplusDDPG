from PIL import Image
import numpy as np

# 이미지 파일 경로
image_path = "/home/dmsgv1/turtlebot3_drlnav/src/turtlebot3_drl/vae/images/depth1/10000.png"

# 이미지 열기
image = Image.open(image_path)

# 이미지를 numpy 배열로 변환
image_array = np.array(image)

# 각 픽셀 값 출력
print(image_array, image_array.shape, image_array.min(), image_array.max())