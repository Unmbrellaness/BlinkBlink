"""
EAR眨眼检测器 - 使用眼睛纵横比检测眨眼
"""
import numpy as np
from scipy.spatial import distance
from typing import Tuple, List, Optional
import threading
import time
from collections import deque
import cv2


class EARCalculator:
    """眼睛纵横比（EAR）计算器"""

    # 面部关键点索引
    # 眼睛关键点（68点模型）
    LEFT_EYE = list(range(42, 48))  # 左眼
    RIGHT_EYE = list(range(36, 42))  # 右眼

    # 眼睛关键点（5点模型）
    LEFT_EYE_5 = [0, 1, 2, 3, 4]  # 左眼
    RIGHT_EYE_5 = [5, 6, 7, 8, 9]  # 右眼

    def __init__(self, eye_model: str = "5point"):
        """
        初始化EAR计算器

        Args:
            eye_model: "5point" 或 "68point"
        """
        self.eye_model = eye_model
        self.ear_threshold = 0.21  # EAR阈值，低于此值认为眼睛闭合
        self.ear_consecutive_frames = 2  # 连续帧数阈值
        self.blink_duration_threshold = 0.10  # 眨眼持续时间阈值（秒）

    def set_thresholds(self, ear_threshold: float = 0.21, consecutive_frames: int = 2):
        """设置检测阈值"""
        self.ear_threshold = ear_threshold
        self.ear_consecutive_frames = consecutive_frames

    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """
        计算眼睛纵横比（EAR）

        Args:
            eye_points: 眼睛关键点坐标，形状为 (n, 2)

        Returns:
            float: EAR值
        """
        # 计算眼睛的水平距离（眼角到眼角）
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])

        # 计算眼睛的垂直距离（眼睛上下边缘）
        C = distance.euclidean(eye_points[0], eye_points[3])

        # 计算EAR
        ear = (A + B) / (2.0 * C)

        return ear

    def calculate_eye_aspect_ratio(self, landmarks, detector_type: str = "mediapipe") -> Tuple[float, float]:
        """
        计算双眼的EAR

        Args:
            landmarks: 面部关键点
            detector_type: "mediapipe" 或 "dlib"

        Returns:
            Tuple[float, float]: (左眼EAR, 右眼EAR)
        """
        if detector_type == "mediapipe":
            # MediaPipe 5点眼睛模型
            if self.eye_model == "5point":
                # 左眼关键点
                left_eye = np.array([
                    [landmarks[33].x, landmarks[33].y],
                    [landmarks[160].x, landmarks[160].y],
                    [landmarks[158].x, landmarks[158].y],
                    [landmarks[133].x, landmarks[133].y],
                    [landmarks[153].x, landmarks[153].y],
                    [landmarks[144].x, landmarks[144].y]
                ])

                # 右眼关键点
                right_eye = np.array([
                    [landmarks[362].x, landmarks[362].y],
                    [landmarks[385].x, landmarks[385].y],
                    [landmarks[387].x, landmarks[387].y],
                    [landmarks[263].x, landmarks[263].y],
                    [landmarks[373].x, landmarks[373].y],
                    [landmarks[380].x, landmarks[380].y]
                ])
            else:
                # 68点模型
                left_eye = np.array([
                    [landmarks[36].x, landmarks[36].y],
                    [landmarks[37].x, landmarks[37].y],
                    [landmarks[38].x, landmarks[38].y],
                    [landmarks[39].x, landmarks[39].y],
                    [landmarks[40].x, landmarks[40].y],
                    [landmarks[41].x, landmarks[41].y]
                ])

                right_eye = np.array([
                    [landmarks[42].x, landmarks[42].y],
                    [landmarks[43].x, landmarks[43].y],
                    [landmarks[44].x, landmarks[44].y],
                    [landmarks[45].x, landmarks[45].y],
                    [landmarks[46].x, landmarks[46].y],
                    [landmarks[47].x, landmarks[47].y]
                ])

        elif detector_type == "dlib":
            # Dlib 68点模型
            left_eye = np.array([
                [landmarks[36].x, landmarks[36].y],
                [landmarks[37].x, landmarks[37].y],
                [landmarks[38].x, landmarks[38].y],
                [landmarks[39].x, landmarks[39].y],
                [landmarks[40].x, landmarks[40].y],
                [landmarks[41].x, landmarks[41].y]
            ])

            right_eye = np.array([
                [landmarks[42].x, landmarks[42].y],
                [landmarks[43].x, landmarks[43].y],
                [landmarks[44].x, landmarks[44].y],
                [landmarks[45].x, landmarks[45].y],
                [landmarks[46].x, landmarks[46].y],
                [landmarks[47].x, landmarks[47].y]
            ])
        else:
            raise ValueError(f"不支持的检测器类型: {detector_type}")

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        # 返回双眼平均EAR
        return (left_ear + right_ear) / 2.0, left_ear

    def is_blinking(self, ear: float) -> bool:
        """判断是否正在眨眼（眼睛闭合）"""
        return ear < self.ear_threshold


class AdaptiveEARCalculator(EARCalculator):
    """
    自适应EAR计算器 - 动态调整眨眼检测阈值

    核心思想：
    1. 使用滑动窗口记录过去N秒的EAR值
    2. 假设用户在这段时间内至少有一次自然睁眼
    3. 取窗口内EAR最大值的某个比例(如0.7)作为眨眼判定阈值
    4. 添加上下界，防止阈值过小或过大

    优势：
    - 适应不同用户的眼睛大小
    - 适应不同光照条件
    - 解决非正向面对摄像头时的EAR偏差问题
    """

    def __init__(self, eye_model: str = "5point",
                 window_seconds: float = 5.0,
                 threshold_ratio: float = 0.7,
                 min_threshold: float = 0.15,
                 max_threshold: float = 0.35,
                 fps: int = 30):
        """
        初始化自适应EAR计算器

        Args:
            eye_model: "5point" 或 "68point"
            window_seconds: 滑动窗口大小（秒）
            threshold_ratio: 阈值比例（最大值 * ratio = 眨眼阈值）
            min_threshold: 阈值下界
            max_threshold: 阈值上界
            fps: 帧率
        """
        super().__init__(eye_model)

        # 自适应阈值参数
        self.window_seconds = window_seconds
        self.threshold_ratio = threshold_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # 滑动窗口
        window_size = int(window_seconds * fps)
        self.ear_window = deque(maxlen=window_size)  # EAR值队列
        self.ear_timestamps = deque(maxlen=window_size)  # 对应时间戳

        # 当前动态阈值
        self.dynamic_threshold = 0.21  # 初始值
        self.is_initialized = False  # 窗口是否已填满

    def update(self, ear: float, timestamp: float = None) -> float:
        """
        更新EAR值并计算动态阈值

        Args:
            ear: 当前EAR值
            timestamp: 时间戳

        Returns:
            float: 当前的动态眨眼阈值
        """
        if timestamp is None:
            timestamp = time.time()

        # 过滤异常值（过小可能是检测错误）
        if ear > 0.05:  # 合理EAR值
            self.ear_window.append(ear)
            self.ear_timestamps.append(timestamp)

        # 计算动态阈值
        if len(self.ear_window) >= 10:  # 至少10帧才开始计算
            max_ear = max(self.ear_window)
            self.dynamic_threshold = max_ear * self.threshold_ratio
            self.dynamic_threshold = max(self.min_threshold,
                                        min(self.max_threshold, self.dynamic_threshold))
            self.is_initialized = True
        else:
            # 窗口未满，使用初始阈值
            self.dynamic_threshold = self.ear_threshold
            self.is_initialized = False

        return self.dynamic_threshold

    def is_blinking(self, ear: float) -> bool:
        """使用动态阈值判断是否眨眼"""
        # 如果窗口未初始化，使用固定阈值
        if not self.is_initialized:
            return ear < self.ear_threshold
        return ear < self.dynamic_threshold

    def get_threshold_info(self) -> dict:
        """获取阈值信息"""
        info = {
            'dynamic_threshold': self.dynamic_threshold,
            'is_initialized': self.is_initialized,
            'window_size': len(self.ear_window),
            'max_ear_in_window': max(self.ear_window) if self.ear_window else 0.0,
            'min_ear_in_window': min(self.ear_window) if self.ear_window else 0.0,
            'threshold_ratio': self.threshold_ratio,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
        }
        return info

    def reset(self):
        """重置滑动窗口"""
        self.ear_window.clear()
        self.ear_timestamps.clear()
        self.dynamic_threshold = self.ear_threshold
        self.is_initialized = False

    def set_threshold_bounds(self, min_t: float = None, max_t: float = None,
                             ratio: float = None):
        """设置阈值边界"""
        if min_t is not None:
            self.min_threshold = min_t
        if max_t is not None:
            self.max_threshold = max_t
        if ratio is not None:
            self.threshold_ratio = ratio


class BlinkDetector:
    """眨眼检测器 - 封装眨眼检测逻辑"""

    def __init__(self, ear_calculator: EARCalculator = None,
                 use_adaptive_threshold: bool = False,
                 adaptive_config: dict = None):
        """
        初始化眨眼检测器

        Args:
            ear_calculator: EAR计算器实例
            use_adaptive_threshold: 是否使用自适应阈值
            adaptive_config: 自适应阈值配置 {'window_seconds': 5.0, 'threshold_ratio': 0.7, ...}
        """
        if use_adaptive_threshold and adaptive_config:
            self.ear_calculator = AdaptiveEARCalculator(**adaptive_config)
        else:
            self.ear_calculator = ear_calculator or EARCalculator()

        self.use_adaptive_threshold = use_adaptive_threshold

        # 眨眼检测状态
        self.blink_count = 0
        self.blink_times = []  # 眨眼时间戳列表
        self.current_blink_start = None
        self.is_currently_blinking = False
        self.last_blink_time = None  # 记录上次眨眼时间戳

        # 统计
        self.total_blinks = 0
        self.last_reset_time = time.time()
        self.blinks_per_minute = 0.0

    def reset(self):
        """重置检测状态"""
        self.blink_count = 0
        self.blink_times = []
        self.current_blink_start = None
        self.is_currently_blinking = False
        self.last_blink_time = None  # 新增
        self.total_blinks = 0
        self.last_reset_time = time.time()

        # 重置自适应阈值窗口
        if self.use_adaptive_threshold and hasattr(self.ear_calculator, 'reset'):
            self.ear_calculator.reset()

    def update(self, ear: float, timestamp: float = None) -> dict:
        """
        更新眨眼状态

        Args:
            ear: 当前EAR值
            timestamp: 时间戳

        Returns:
            dict: 检测结果 {'blink_detected': bool, 'is_blinking': bool, 'ear': float}
        """
        if timestamp is None:
            timestamp = time.time()

        # 如果使用自适应阈值，先更新阈值
        if self.use_adaptive_threshold and hasattr(self.ear_calculator, 'update'):
            self.ear_calculator.update(ear, timestamp)

        is_blinking = self.ear_calculator.is_blinking(ear)
        result = {
            'blink_detected': False,
            'is_blinking': is_blinking,
            'ear': ear,
            'timestamp': timestamp
        }

        # 检测眨眼开始
        if is_blinking and not self.is_currently_blinking:
            self.is_currently_blinking = True
            self.current_blink_start = timestamp

        # 检测眨眼结束（一次完整的眨眼）
        elif not is_blinking and self.is_currently_blinking:
            blink_duration = timestamp - self.current_blink_start

            # 只有持续时间足够的才算有效眨眼
            if blink_duration >= self.ear_calculator.blink_duration_threshold:
                # 间隔过滤：两次眨眼至少相隔1秒，防止误检激增
                if self.last_blink_time is not None and timestamp - self.last_blink_time < 1.0:
                    self.is_currently_blinking = False
                    self.current_blink_start = None
                    # print("眨眼间隔过滤：两次眨眼至少相隔1秒，防止误检激增")
                    return result  # 直接丢弃，blink_count 不增加
                
                self.blink_count += 1
                self.total_blinks += 1
                self.blink_times.append(timestamp)
                self.last_blink_time = timestamp  # 更新上次眨眼时间
                result['blink_detected'] = True
                result['blink_duration'] = blink_duration
                # print("眨眼检测：眨眼次数增加:", self.blink_count)

            self.is_currently_blinking = False
            self.current_blink_start = None

        return result

    def get_blink_rate(self, window_seconds: float = 60.0) -> float:
        """
        计算眨眼频率（每分钟眨眼次数）

        Args:
            window_seconds: 计算窗口大小（秒）

        Returns:
            float: 每分钟眨眼次数
        """
        current_time = time.time()

        # 清理过期的眨眼记录
        self.blink_times = [t for t in self.blink_times
                          if current_time - t < window_seconds]

        recent_blinks = len(self.blink_times)

        # 计算每分钟眨眼次数
        blink_rate = (recent_blinks / window_seconds) * 60.0

        return blink_rate

    def get_statistics(self) -> dict:
        """获取检测统计信息"""
        stats = {
            'total_blinks': self.total_blinks,
            'current_blink_count': self.blink_count,
            'blink_rate_per_minute': self.get_blink_rate(),
            'is_currently_blinking': self.is_currently_blinking,
            'ear_threshold': self.ear_calculator.ear_threshold,
            'use_adaptive_threshold': self.use_adaptive_threshold
        }

        # 如果使用自适应阈值，添加详细信息
        if self.use_adaptive_threshold and hasattr(self.ear_calculator, 'get_threshold_info'):
            stats['threshold_info'] = self.ear_calculator.get_threshold_info()

        return stats


class AsyncBlinkDetector:
    """异步眨眼检测器 - 用于后台运行"""

    def __init__(self, camera_manager, ear_calculator: EARCalculator = None):
        """
        初始化异步眨眼检测器

        Args:
            camera_manager: 摄像头管理器实例
            ear_calculator: EAR计算器实例
        """
        self.camera_manager = camera_manager
        self.ear_calculator = ear_calculator or EARCalculator()
        self.blink_detector = BlinkDetector(self.ear_calculator)

        # 控制标志
        self.running = False
        self.thread = None

        # 检测间隔
        self.detection_interval = 0.02  # 约50FPS

        # 回调函数
        self.on_blink_callback = None
        self.on_status_update_callback = None

        # 统计
        self.frame_count = 0
        self.detection_latency = 0.0

    def start(self, async_mode: bool = True):
        """启动检测"""
        if self.running:
            return

        self.running = True

        if async_mode:
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
        else:
            self._detection_loop()

    def stop(self):
        """停止检测"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def set_blink_callback(self, callback):
        """设置眨眼回调函数"""
        self.on_blink_callback = callback

    def set_status_callback(self, callback):
        """设置状态更新回调函数"""
        self.on_status_update_callback = callback

    def _detection_loop(self):
        """检测主循环"""
        import mediapipe as mp

        # 初始化MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        try:
            while self.running:
                start_time = time.time()

                # 从摄像头获取帧
                frame_data = self.camera_manager.get_latest_frame("blink_detector")

                if frame_data is None:
                    time.sleep(0.1)
                    continue

                frame = frame_data['frame']

                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测面部关键点
                results = face_mesh.process(frame_rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                    # 计算EAR
                    avg_ear, _ = self.ear_calculator.calculate_eye_aspect_ratio(
                        landmarks, "mediapipe"
                    )

                    # 更新检测状态
                    result = self.blink_detector.update(
                        avg_ear,
                        frame_data.get('timestamp', start_time)
                    )

                    # 触发眨眼回调
                    if result['blink_detected'] and self.on_blink_callback:
                        self.on_blink_callback(result)

                    # 触发状态更新回调
                    if self.on_status_update_callback:
                        self.on_status_update_callback({
                            **result,
                            'statistics': self.blink_detector.get_statistics(),
                            'frame_count': self.frame_count
                        })

                self.frame_count += 1

                # 控制检测频率
                elapsed = time.time() - start_time
                sleep_time = max(0, self.detection_interval - elapsed)
                time.sleep(sleep_time)

                self.detection_latency = time.time() - start_time

        except Exception as e:
            print(f"眨眼检测循环异常: {e}")
        finally:
            face_mesh.close()

    def get_statistics(self) -> dict:
        """获取检测统计信息"""
        stats = self.blink_detector.get_statistics()
        stats['frame_count'] = self.frame_count
        stats['detection_latency'] = self.detection_latency
        return stats
