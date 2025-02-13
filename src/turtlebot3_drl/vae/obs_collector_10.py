import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import json
import os
from collections import deque
import csv

class ObsCollector(Node):
    def __init__(self):
        super().__init__('obs_collector')
        self.rgb_sub = self.create_subscription(Image, '/camera1/camera1/image_raw', self.rgb_callback, 10)
        self.depth_sub1 = self.create_subscription(Image, '/depth_camera1/depth_camera1/depth/image_raw', self.depth_callback1, 10)
        self.depth_sub2 = self.create_subscription(Image, '/depth_camera2/depth_camera2/depth/image_raw', self.depth_callback2, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.pose_sub = self.create_subscription(Odometry, '/odom', self.pose_callback, 10)

        self.rgb_image = None
        self.depth_image1 = None
        self.depth_image2 = None
        self.agent_pose = None
        self.lidar = None

        self.rgb_path = 'images/rgb'
        self.depth1_path = 'images/depth1'
        self.depth2_path = 'images/depth2'
        self.pose_path = 'images/pose.json'
        self.lidar_path = 'images/lidar'

        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.depth1_path, exist_ok=True)
        os.makedirs(self.depth2_path, exist_ok=True)
        os.makedirs(self.lidar_path, exist_ok=True)

        self.global_index = 10000
        self.pose_data = []  # Pose 데이터를 저장할 리스트 초기화

    def rgb_callback(self, msg):
        try:
            self.rgb_image = self.convert_image(msg, dtype=np.uint8)
            filename = f'{self.global_index:04d}.png'
            self.save_image(self.rgb_image, self.rgb_path, filename, is_depth=False)
            self.get_logger().info(f'RGB image received and saved as {filename}')
            self.check_and_save_pose_data()
        except Exception as e:
            self.get_logger().error(f'Failed to process RGB image: {e}')

    def depth_callback1(self, msg):
        try:
            self.depth_image1 = self.convert_image(msg, dtype=np.float32)
            filename = f'{self.global_index:04d}.npy'  # 확장자를 .npy로 변경
            self.save_image(self.depth_image1, self.depth1_path, filename, is_depth=True)
            self.get_logger().info(f'Depth image1 received and saved as {filename}')
            self.check_and_save_pose_data()
        except Exception as e:
            self.get_logger().error(f'Failed to process Depth image1: {e}')

    def depth_callback2(self, msg):
        try:
            self.depth_image2 = self.convert_image(msg, dtype=np.float32)
            filename = f'{self.global_index:04d}.npy'  # 확장자를 .npy로 변경
            self.save_image(self.depth_image2, self.depth2_path, filename, is_depth=True)
            self.get_logger().info(f'Depth image2 received and saved as {filename}')
            self.check_and_save_pose_data()
        except Exception as e:
            self.get_logger().error(f'Failed to process Depth image2: {e}')

    def pose_callback(self, msg):
        try:
            self.agent_pose = {
                'position': [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ],
                'orientation': [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w
                ]
            }
            self.check_and_save_pose_data()
        except Exception as e:
            self.get_logger().error(f'Failed to process Pose data: {e}')
        
    def scan_callback(self, msg):
        self.lidar = msg.ranges
        self.save_lidar_data(self.lidar, self.lidar_path)
        self.check_and_save_pose_data()

    def check_and_save_pose_data(self):
        if self.rgb_image is not None and self.depth_image1 is not None and self.depth_image2 is not None and self.agent_pose is not None and self.lidar is not None:
            self.pose_data.append(self.agent_pose)
            self.save_pose(self.pose_data, self.pose_path)
            self.rgb_image = None  # 저장 후 초기화
            self.depth_image1 = None  # 저장 후 초기화
            self.depth_image2 = None  # 저장 후 초기화
            self.agent_pose = None  # 저장 후 초기화
            self.lidar = None  # 저장 후 초기화
            self.global_index += 1

    def convert_image(self, msg, dtype):
        """
        ROS Image 메시지를 OpenCV 이미지로 변환합니다.
        :param msg: sensor_msgs.msg.Image 메시지
        :param dtype: 변환할 데이터 타입 (예: np.uint8, np.float32)
        :return: 변환된 OpenCV 이미지
        """
        height, width = msg.height, msg.width
        encoding = msg.encoding
        if encoding in ['rgb8', 'bgr8']:
            channels = 3
        elif encoding in ['mono8', '32FC1']:
            channels = 1
        else:
            self.get_logger().warning(f'Unsupported encoding: {encoding}, defaulting to 1 channel')
            channels = 1
        img = np.frombuffer(msg.data, dtype=dtype).reshape(height, width, channels)
        
        # RGB to BGR 변환 (OpenCV 호환성 위해)
        if encoding == 'rgb8':
            img = img[..., ::-1]  # RGB to BGR
        
        # 디버깅: 이미지 형태 출력
        self.get_logger().debug(f'Converted image shape: {img.shape}')
        
        return img

    def save_image(self, image, path, filename, is_depth=False):
        """
        이미지를 파일로 저장합니다.
        :param image: 저장할 이미지 (numpy 배열)
        :param path: 저장할 경로
        :param filename: 파일 이름
        :param is_depth: Depth 이미지인지 여부
        """
        if is_depth:
            # 깊이 이미지를 .npy 파일로 저장하여 float32 값을 보존
            np.save(os.path.join(path, filename), image)
        else:
            cv2.imwrite(os.path.join(path, filename), image)

    def save_pose(self, pose_data, path):
        """
        Pose 데이터를 JSON 파일로 저장합니다.
        :param pose_data: Pose 데이터 리스트
        :param path: 저장할 경로
        """
        try:
            with open(path, 'w') as f:
                json.dump(pose_data, f, indent=4)
            self.get_logger().info(f'Pose data saved to {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save pose data: {e}')
        print(f'Total pose data saved: {len(self.pose_data)}')

    def save_lidar_data(self, lidar_data, path):
        """
        LiDAR 데이터를 파일에 저장합니다.
        :param lidar_data: 리스트 형태의 LiDAR 데이터
        :param path: 데이터를 저장할 폴더 경로
        """
        
        # 파일 이름 생성 (예: "0001.csv")
        filename = f'{self.global_index:04d}.csv'
        full_path = os.path.join(path, filename)
        
        # CSV 파일로 저장
        with open(full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for data_point in lidar_data:
                writer.writerow([data_point])

def main(args=None):
    rclpy.init(args=args)
    node = ObsCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
