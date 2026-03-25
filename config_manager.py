"""
配置管理器 - 管理干眼检测器的所有配置项
支持保存/加载配置，支持默认配置
"""
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import threading


# =============================================================================
# 检测相关配置
# =============================================================================

@dataclass
class DetectionSettings:
    """
    眨眼检测相关配置
    
    说明：
    - use_adaptive_threshold: 是否启用自适应阈值
      启用后会自动根据用户眨眼习惯调整阈值
    - adaptive_window_seconds: 自适应阈值滑动窗口大小（秒）
      窗口越大阈值越稳定，但适应速度越慢
    - adaptive_threshold_ratio: 自适应阈值比例
      窗口内EAR最大值 * 此比例 = 眨眼判定阈值
    - adaptive_min_threshold: 自适应阈值下界
      防止阈值过小导致漏检
    - adaptive_max_threshold: 自适应阈值上界
      防止阈值过大导致误检
    """
    # 自适应阈值配置
    use_adaptive_threshold: bool = True  # 是否启用自适应阈值
    adaptive_window_seconds: float = 5.0  # 滑动窗口大小 (秒)
    adaptive_threshold_ratio: float = 0.7  # 阈值比例 (0.5-0.9)
    adaptive_min_threshold: float = 0.15  # 阈值下界
    adaptive_max_threshold: float = 0.35  # 阈值上界


@dataclass
class MediaPipeSettings:
    """
    MediaPipe面部检测相关配置
    
    说明：
    - min_detection_confidence: 人脸检测最小置信度
      低于此值认为未检测到人脸
      调低可以更容易检测到人脸，但可能误检
    - min_tracking_confidence: 关键点跟踪最小置信度
      调低可以让跟踪更稳定，但可能不够精确
    - refine_landmarks: 是否精炼关键点
      精炼眼睛和嘴唇关键点，精确度更高但消耗更多资源
    """
    min_detection_confidence: float = 0.5  # 人脸检测置信度 (0.3-0.8)
    min_tracking_confidence: float = 0.3  # 关键点跟踪置信度 (0.2-0.7)
    refine_landmarks: bool = True  # 是否精炼眼睛和嘴唇关键点

# =============================================================================
# 提醒相关配置
# =============================================================================

@dataclass
class AlertSettings:
    """
    系统提醒相关配置
    
    说明：
    - enabled: 是否启用提醒功能
    - use_system_notification: 是否使用Windows系统通知
    - use_in_app_notification: 是否在应用内显示通知
    - sound_enabled: 是否播放提醒声音
    - voice_enabled: 是否启用语音播报
    - alert_interval: 提醒间隔（秒）
    """
    enabled: bool = True  # 是否启用提醒功能
    use_system_notification: bool = True  # 是否使用Windows系统通知
    use_in_app_notification: bool = True  # 是否在应用内显示通知
    sound_enabled: bool = True  # 是否播放提醒声音
    voice_enabled: bool = True  # 是否启用语音播报
    alert_interval: int = 30  # 提醒间隔（秒）


# =============================================================================
# 摄像头相关配置
# =============================================================================

@dataclass
class CameraSettings:
    """
    摄像头相关配置
    
    说明：
    - camera_index: 摄像头索引
      0通常是默认摄像头，1是第二个摄像头，以此类推
    - buffer_size: 帧缓冲区大小
      越大越流畅，但延迟越高
    - target_fps: 目标帧率
      摄像头采集帧率，不是检测帧率
    """
    camera_index: int = 0  # 摄像头索引 (0-9)
    buffer_size: int = 10  # 帧缓冲区大小 (5-30)
    target_fps: int = 15  # 目标帧率 (15-60)


# =============================================================================
# 应用主配置
# =============================================================================

@dataclass
class AppConfig:
    """
    应用主配置类
    
    包含所有配置组，使用dataclass的field工厂函数创建默认值
    """
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    mediapipe: MediaPipeSettings = field(default_factory=MediaPipeSettings)
    alert: AlertSettings = field(default_factory=AlertSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    
    # 通用配置
    auto_start: bool = False  # 启动时自动开始检测
    minimize_to_tray: bool = True  # 最小化到系统托盘
    stats_retention_days: int = 7  # 统计数据保留天数


class ConfigManager:
    """配置管理器"""
    
    CONFIG_FILE = "config.json"
    
    # 配置描述字典，用于UI显示
    CONFIG_DESCRIPTIONS = {
        # MediaPipe配置
        'mediapipe.min_detection_confidence': '人脸检测置信度：人脸检测的最小置信度阈值',
        'mediapipe.min_tracking_confidence': '关键点跟踪置信度：面部关键点跟踪的最小置信度',
        'mediapipe.refine_landmarks': '精炼关键点：是否精炼眼睛和嘴唇关键点（更精确但更慢）',
        # 提醒配置
        'alert.enabled': '启用提醒：是否启用眼睛疲劳提醒功能',
        'alert.use_system_notification': '系统通知：是否使用Windows系统通知',
        'alert.use_in_app_notification': '应用内通知：是否在窗口内显示通知',
        'alert.sound_enabled': '提醒声音：是否播放声音提醒',
        # 摄像头配置
        'camera.camera_index': '摄像头索引：选择使用的摄像头（0通常是默认摄像头）',
        'camera.buffer_size': '缓冲区大小：摄像头帧缓冲区大小',
        'camera.target_fps': '目标帧率：摄像头采集的目标帧率',
    }
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = config_dir
        self.config_path = os.path.join(config_dir, self.CONFIG_FILE)
        self.lock = threading.Lock()
        self.config = AppConfig()
        self.load()
    
    def get_description(self, key: str) -> str:
        """获取配置项的中文描述"""
        return self.CONFIG_DESCRIPTIONS.get(key, '')
    
    def load(self) -> bool:
        if not os.path.exists(self.config_path):
            print(f"配置文件不存在，将使用默认配置: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_config(data)
            print(f"配置已从文件加载: {self.config_path}")
            return True
        except Exception as e:
            print(f"加载配置时出错: {e}")
            return False
    
    def _apply_config(self, data: Dict[str, Any]) -> None:
        # 检测配置
        if 'detection' in data:
            d = data['detection']
            # 自适应阈值配置
            self.config.detection.use_adaptive_threshold = d.get('use_adaptive_threshold', self.config.detection.use_adaptive_threshold)
            self.config.detection.adaptive_window_seconds = d.get('adaptive_window_seconds', self.config.detection.adaptive_window_seconds)
            self.config.detection.adaptive_threshold_ratio = d.get('adaptive_threshold_ratio', self.config.detection.adaptive_threshold_ratio)
            self.config.detection.adaptive_min_threshold = d.get('adaptive_min_threshold', self.config.detection.adaptive_min_threshold)
            self.config.detection.adaptive_max_threshold = d.get('adaptive_max_threshold', self.config.detection.adaptive_max_threshold)
        
        # MediaPipe配置
        if 'mediapipe' in data:
            m = data['mediapipe']
            self.config.mediapipe.min_detection_confidence = m.get('min_detection_confidence', self.config.mediapipe.min_detection_confidence)
            self.config.mediapipe.min_tracking_confidence = m.get('min_tracking_confidence', self.config.mediapipe.min_tracking_confidence)
            self.config.mediapipe.refine_landmarks = m.get('refine_landmarks', self.config.mediapipe.refine_landmarks)
         
        # 提醒配置
        if 'alert' in data:
            a = data['alert']
            self.config.alert.enabled = a.get('enabled', self.config.alert.enabled)
            self.config.alert.use_system_notification = a.get('use_system_notification', self.config.alert.use_system_notification)
            self.config.alert.use_in_app_notification = a.get('use_in_app_notification', self.config.alert.use_in_app_notification)
            self.config.alert.sound_enabled = a.get('sound_enabled', self.config.alert.sound_enabled)
            self.config.alert.voice_enabled = a.get('voice_enabled', self.config.alert.voice_enabled)
            self.config.alert.alert_interval = a.get('alert_interval', self.config.alert.alert_interval)
        
        # 摄像头配置
        if 'camera' in data:
            c = data['camera']
            self.config.camera.camera_index = c.get('camera_index', self.config.camera.camera_index)
            self.config.camera.buffer_size = c.get('buffer_size', self.config.camera.buffer_size)
            self.config.camera.target_fps = c.get('target_fps', self.config.camera.target_fps)
        
        # 通用配置
        self.config.auto_start = data.get('auto_start', self.config.auto_start)
        self.config.minimize_to_tray = data.get('minimize_to_tray', self.config.minimize_to_tray)
        self.config.stats_retention_days = data.get('stats_retention_days', self.config.stats_retention_days)
    
    def save(self) -> bool:
        with self.lock:
            try:
                data = self._to_dict()
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"配置已保存: {self.config_path}")
                return True
            except Exception as e:
                print(f"保存配置时出错: {e}")
                return False
    
    def _to_dict(self) -> Dict[str, Any]:
        return {
            'detection': {
                # 自适应阈值配置
                'use_adaptive_threshold': self.config.detection.use_adaptive_threshold,
                'adaptive_window_seconds': self.config.detection.adaptive_window_seconds,
                'adaptive_threshold_ratio': self.config.detection.adaptive_threshold_ratio,
                'adaptive_min_threshold': self.config.detection.adaptive_min_threshold,
                'adaptive_max_threshold': self.config.detection.adaptive_max_threshold
            },
            'mediapipe': {
                'min_detection_confidence': self.config.mediapipe.min_detection_confidence,
                'min_tracking_confidence': self.config.mediapipe.min_tracking_confidence,
                'refine_landmarks': self.config.mediapipe.refine_landmarks
            },
            'alert': {
                'enabled': self.config.alert.enabled,
                'use_system_notification': self.config.alert.use_system_notification,
                'use_in_app_notification': self.config.alert.use_in_app_notification,
                'sound_enabled': self.config.alert.sound_enabled,
                'voice_enabled': self.config.alert.voice_enabled,
                'alert_interval': self.config.alert.alert_interval
            },
            'camera': {
                'camera_index': self.config.camera.camera_index,
                'buffer_size': self.config.camera.buffer_size,
                'target_fps': self.config.camera.target_fps
            },
            'auto_start': self.config.auto_start,
            'minimize_to_tray': self.config.minimize_to_tray,
            'stats_retention_days': self.config.stats_retention_days
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            elif hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
            if value is None:
                return default
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """设置配置值"""
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if isinstance(target, dict):
                if k not in target:
                    return False
                target = target[k]
            elif hasattr(target, k):
                target = getattr(target, k)
            else:
                return False
        final_key = keys[-1]
        if isinstance(target, dict):
            target[final_key] = value
        elif hasattr(target, final_key):
            setattr(target, final_key, value)
        else:
            return False
        return True
    
    def reset_to_defaults(self) -> None:
        """重置为默认配置"""
        self.config = AppConfig()
        self.save()
    
    def export_config(self, path: str) -> bool:
        """导出配置到文件"""
        try:
            data = self._to_dict()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return True
        except Exception as e:
            print(f"导出配置时出错: {e}")
            return False
    
    def import_config(self, path: str) -> bool:
        """从文件导入配置"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._apply_config(data)
            self.save()
            return True
        except Exception as e:
            print(f"导入配置时出错: {e}")
            return False
