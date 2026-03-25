"""
摄像头资源管理器 - 简化架构
全局Frame Provider进程持续抓取帧，所有服务直接从缓冲区获取最新帧
移除状态机，允许多个服务并发访问帧数据
"""
import cv2
import threading
import time
from collections import deque

class FrameProvider:
    """全局帧提供者 - 持续抓取帧并维护缓冲区"""

    def __init__(self, camera_index=0, buffer_size=10, fps=30):
        self.camera_index = camera_index
        self.buffer_size = buffer_size
        self.fps = fps
        self.frame_interval = 1.0 / fps

        # 环形缓冲区存储帧
        self.frame_buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        # 线程控制
        self.running = False
        self.thread = None
        self.cap = None

        # 统计信息
        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_capture_time = 0

        print(f"帧提供者已初始化 - 缓冲区大小: {buffer_size}, 目标FPS: {fps}")

    def start(self):
        """启动帧抓取线程"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("帧提供者已启动")

    def stop(self):
        """停止帧抓取线程"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None

        print("帧提供者已停止")

    def _capture_loop(self):
        """帧抓取主循环"""
        print("开始帧抓取循环...")

        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                print("无法打开摄像头设备")
                self.running = False
                return

            # 设置摄像头参数
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            print("摄像头初始化成功，开始抓取帧")

            while self.running:
                start_time = time.time()

                # 抓取帧
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # 添加时间戳
                    timestamp = time.time()

                    # 存入缓冲区
                    with self.buffer_lock:
                        self.frame_buffer.append({
                            'frame': frame.copy(),
                            'timestamp': timestamp,
                            'frame_id': self.frames_captured
                        })

                    self.frames_captured += 1

                    # 限制缓冲区大小
                    if len(self.frame_buffer) > self.buffer_size:
                        self.frames_dropped += 1
                else:
                    print("帧抓取失败")
                    time.sleep(0.1)
                    continue

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.frame_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"帧抓取循环异常: {e}")
        finally:
            self.running = False
            if self.cap:
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None

    def get_latest_frame(self, timeout=1.0):
        """
        获取最新的帧

        Returns:
            dict: {'frame': frame, 'timestamp': timestamp, 'frame_id': frame_id} or None
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.buffer_lock:
                if self.frame_buffer:
                    # 返回最新的帧（不移除）
                    return self.frame_buffer[-1].copy()
            time.sleep(0.01)

        return None

    def get_frame_buffer_info(self):
        """获取缓冲区信息"""
        with self.buffer_lock:
            return {
                'buffer_size': len(self.frame_buffer),
                'max_buffer_size': self.buffer_size,
                'frames_captured': self.frames_captured,
                'frames_dropped': self.frames_dropped,
                'is_running': self.running
            }

class CameraManager:
    """摄像头资源管理器（单例模式）- 简化架构，直接提供帧访问"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CameraManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_manager=None):
        if self._initialized:
            return

        # 如果提供了配置管理器，从配置读取参数
        if config_manager is not None:
            camera_index = config_manager.get('camera.camera_index', 0)
            buffer_size = config_manager.get('camera.buffer_size', 10)
            target_fps = config_manager.get('camera.target_fps', 30)
            self.frame_provider = FrameProvider(
                camera_index=camera_index,
                buffer_size=buffer_size,
                fps=target_fps
            )
        else:
            self.frame_provider = FrameProvider()
        self._initialized = True

        # 启动帧提供者
        self.frame_provider.start()

        print("摄像头资源管理器已初始化（简化架构）")

    def get_latest_frame(self, service_name="unknown", timeout=1.0):
        """
        直接获取最新的帧（所有服务都可以并发访问）

        Args:
            service_name: 调用服务的名称，用于调试
            timeout: 超时时间（秒）

        Returns:
            dict: {'frame': frame, 'timestamp': timestamp, 'frame_id': frame_id} or None
        """
        frame_data = self.frame_provider.get_latest_frame(timeout)
        # if frame_data:
            # print(f"服务 '{service_name}' 获取帧 {frame_data['frame_id']}")
        return frame_data

    def get_frame_provider_stats(self):
        """获取帧提供者统计信息"""
        return self.frame_provider.get_frame_buffer_info()

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'frame_provider'):
            self.frame_provider.stop()
        print("CameraManager已清理")

# 创建全局摄像头管理器实例
camera_manager = CameraManager()

# 添加程序退出时的清理
import atexit
atexit.register(camera_manager.cleanup)
