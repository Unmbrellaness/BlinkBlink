#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
干眼检测器主界面 - 基于SiliconUI
"""

import sys, os

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(__file__)

# import sys
import cv2
import time
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QDesktopWidget
from PyQt5.QtGui import QImage, QPixmap, QIcon

# SiliconUI 组件
import siui
from siui.core import SiColor, SiGlobal, Si, GlobalFont
from siui.templates.application.application import SiliconApplication
from siui.components.page import SiPage
from siui.components.slider import SiSliderH
from siui.components.titled_widget_group import SiTitledWidgetGroup
from siui.components.option_card import SiOptionCardPlane
from siui.components.widgets import (
    SiDenseHContainer,
    SiDenseVContainer,
    SiPushButton,
    SiSwitch,
    SiLabel,
)
from siui.components.progress_bar import SiProgressBar

# 导入本地模块
from camera_manager import CameraManager
from eye_blink_detector import EARCalculator, BlinkDetector
from perclos_calculator import EyeFatigueAnalyzer
from config_manager import ConfigManager
from alert_thread import AlertThread


# 获取颜色的辅助函数
def get_color(token):
    """获取SiUI颜色值"""
    return SiGlobal.siui.colors.fromToken(token)


class BlinkDetectionThread(QThread):
    """眨眼检测线程"""
    status_update = pyqtSignal(dict)
    blink_detected = pyqtSignal(dict)

    def __init__(self, camera_manager, config_manager):
        super().__init__()
        self.camera_manager = camera_manager
        self.config_manager = config_manager
        self.running = False
        
        # 从配置读取EAR参数
        ear_threshold = 0.21
        blink_duration = 0.05
        consecutive_frames = 1
        detection_fps = 50
        
        # 检查是否启用自适应阈值
        use_adaptive = self.config_manager.get('detection.use_adaptive_threshold', True)
        
        if use_adaptive:
            # 自适应阈值配置
            adaptive_config = {
                'window_seconds': self.config_manager.get('detection.adaptive_window_seconds', 5.0),
                'threshold_ratio': self.config_manager.get('detection.adaptive_threshold_ratio', 0.8),
                'min_threshold': self.config_manager.get('detection.adaptive_min_threshold', 0.15),
                'max_threshold': self.config_manager.get('detection.adaptive_max_threshold', 0.60),
                'fps': detection_fps
            }
            self.blink_detector = BlinkDetector(
                use_adaptive_threshold=True,
                adaptive_config=adaptive_config
            )
            self.ear_calculator = self.blink_detector.ear_calculator  # 保存引用供计算EAR使用
            print(f"[眨眼检测] 启用自适应阈值: 窗口={adaptive_config['window_seconds']}s, "
                  f"比例={adaptive_config['threshold_ratio']}, "
                  f"范围=[{adaptive_config['min_threshold']}, {adaptive_config['max_threshold']}]")
        else:
            # 使用固定阈值
            self.ear_calculator = EARCalculator()
            self.ear_calculator.set_thresholds(
                ear_threshold=ear_threshold,
                consecutive_frames=consecutive_frames
            )
            self.ear_calculator.blink_duration_threshold = blink_duration
            self.blink_detector = BlinkDetector(self.ear_calculator)
            print(f"[眨眼检测] 使用固定阈值: {ear_threshold}")
        
        self.detection_interval = 1.0 / detection_fps

    def run(self):
        import mediapipe as mp
        
        # 从配置读取MediaPipe参数
        det_confidence = self.config_manager.get('mediapipe.min_detection_confidence', 0.5)
        track_confidence = self.config_manager.get('mediapipe.min_tracking_confidence', 0.3)
        refine_landmarks = self.config_manager.get('mediapipe.refine_landmarks', True)
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=det_confidence,
            min_tracking_confidence=track_confidence,
            static_image_mode=False,
        )

        self.running = True

        while self.running:
            start_time = time.time()
            frame_data = self.camera_manager.get_latest_frame("blink_detector")

            if frame_data is None:
                time.sleep(0.1)
                continue

            frame = frame_data['frame']
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                avg_ear, _ = self.ear_calculator.calculate_eye_aspect_ratio(landmarks, "mediapipe")
                result = self.blink_detector.update(avg_ear, start_time)

                status = {
                    **result,
                    'statistics': self.blink_detector.get_statistics(),
                    'frame': frame
                }
                self.status_update.emit(status)

                if result['blink_detected']:
                    self.blink_detected.emit(result)

            elapsed = time.time() - start_time
            sleep_time = max(0, self.detection_interval - elapsed)
            time.sleep(sleep_time)

        face_mesh.close()

    def stop(self):
        self.running = False
        self.wait()


class MonitorPage(SiPage):
    """监控页面"""

    def __init__(self, app):
        super().__init__(app)
        self.app = app

        self.scroll_container = SiTitledWidgetGroup(self)
        
        card_w, card_h = 180, 140 # 适当放宽

        # 标题
        self.title = SiLabel(self)
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.title.setText("实时眨眼监测")
        self.title.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 32px; font-weight: bold;")

        # 视频卡片
        self.video_card = SiOptionCardPlane(self)
        self.video_card.setTitle("摄像头画面")
        self.video_card.setFixedSize(480, 380)

        self.video_label = SiLabel(self.video_card.body())
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("点击「开始检测」启动摄像头")
        self.video_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_B)}; font-size: 18px;")
        self.video_label.setFixedSize(480, 300)
        
        # 允许 Label 随容器拉伸，而不是固定大小
        # self.video_label.setSizePolicy(Qt.Expanding, Qt.Expanding) 
        self.video_card.body().addWidget(self.video_label)

        # 统计卡片容器
        self.stats_container = SiDenseHContainer(self)
        self.stats_container.setFixedHeight(card_h)
        self.stats_container.setAdjustWidgetsSize(True)

        # 眨眼次数卡片
        self.blink_count_card = SiOptionCardPlane(self)
        self.blink_count_card.setTitle("今日眨眼次数")
        self.blink_count_card.setFixedSize(card_w, card_h)

        self.blink_count_value = SiLabel(self.blink_count_card.body())
        self.blink_count_value.setAlignment(Qt.AlignCenter)
        self.blink_count_value.setText("0")
        self.blink_count_value.setStyleSheet(f"color: {get_color(SiColor.THEME)}; font-size: 56px; font-weight: bold;")
        self.blink_count_value.setFixedSize(120, 50)
        self.blink_count_card.body().addWidget(self.blink_count_value)

        # 眨眼频率卡片
        self.blink_rate_card = SiOptionCardPlane(self)
        self.blink_rate_card.setTitle("眨眼频率 (次/分钟)")
        self.blink_rate_card.setFixedSize(220, card_h)

        self.blink_rate_value = SiLabel(self.blink_rate_card.body())
        self.blink_rate_value.setAlignment(Qt.AlignCenter)
        self.blink_rate_value.setText("0")
        self.blink_rate_value.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 36px; font-weight: bold;")
        self.blink_rate_value.setFixedSize(120, 50)
        self.blink_rate_card.body().addWidget(self.blink_rate_value)

        # 眼睛护眼状态卡片
        self.fatigue_card = SiOptionCardPlane(self)
        self.fatigue_card.setTitle("眼睛护眼状态")
        self.fatigue_card.setFixedSize(180, card_h)
        # 居中
        self.fatigue_card.body().setAlignment(Qt.AlignCenter)
        
        self.fatigue_value = SiLabel(self.fatigue_card.body())
        self.fatigue_value.setAlignment(Qt.AlignCenter)
        self.fatigue_value.setText("良好")
        self.fatigue_value.setStyleSheet("color: #28a745; font-size: 28px; font-weight: bold;")
        self.fatigue_value.setFixedSize(100, 50)
        self.fatigue_card.body().addWidget(self.fatigue_value)

        # 护眼分数卡片
        self.fatigue_score_card = SiOptionCardPlane(self)
        self.fatigue_score_card.setTitle("护眼分数")
        self.fatigue_score_card.setFixedSize(140, card_h)

        self.fatigue_score_value = SiLabel(self.fatigue_score_card.body())
        self.fatigue_score_value.setAlignment(Qt.AlignCenter)
        self.fatigue_score_value.setText("100")
        self.fatigue_score_value.setStyleSheet("color: #28a745; font-size: 32px; font-weight: bold;")
        self.fatigue_score_value.setFixedSize(60, 50)
        self.fatigue_score_card.body().addWidget(self.fatigue_score_value)

        # 添加卡片到容器
        self.stats_container.addWidget(self.blink_count_card)
        self.stats_container.addWidget(self.blink_rate_card)
        self.stats_container.addWidget(self.fatigue_card)
        self.stats_container.addWidget(self.fatigue_score_card)

        # 控制按钮
        self.control_container = SiDenseHContainer(self)
        self.control_container.setFixedHeight(60)
        self.control_container.setAdjustWidgetsSize(True)

        self.start_button = SiPushButton(self)
        self.start_button.setFixedSize(180, 50)
        self.start_button.attachment().setText("开始检测")

        self.reset_button = SiPushButton(self)
        self.reset_button.setFixedSize(140, 50)
        self.reset_button.attachment().setText("重置统计")

        self.control_container.addWidget(self.start_button)
        self.control_container.addWidget(self.reset_button)

        # 添加到滚动容器
        self.scroll_container.addWidget(self.title)
        self.scroll_container.addWidget(self.video_card)
        self.scroll_container.addWidget(self.stats_container)
        self.scroll_container.addWidget(self.control_container)

        self.setAttachment(self.scroll_container)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        self.scroll_container.setFixedWidth(min(w - 128, 1100))
        self.video_card.setFixedWidth(min(w - 128, 1100))

    def update_stats(self, blink_count, blink_rate, perclos, ear, fatigue_level, fatigue_score=0, frame=None):
        """更新统计信息"""
        self.blink_count_value.setText(str(blink_count))
        self.blink_rate_value.setText(f"{blink_rate:.1f}")

        # 更新护眼状态颜色
        eye_health_colors = {
            "优秀": "#00c851",  # 绿色 - 眨眼频率很高
            "良好": "#28a745",  # 绿色 - 眨眼频率正常
            "一般": "#ffc107",  # 黄色 - 眨眼频率偏低
            "需注意": "#fd7e14",  # 橙色 - 眨眼频率过低
            "提醒": "#dc3545"  # 红色 - 眨眼频率严重过低
        }
        color = eye_health_colors.get(fatigue_level, get_color(SiColor.TEXT_A))
        self.fatigue_value.setText(fatigue_level)
        self.fatigue_value.setStyleSheet(f"color: {color}; font-size: 22px; font-weight: bold;")

        # 更新护眼分数
        self.fatigue_score_value.setText(str(fatigue_score))
        self.fatigue_score_value.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold;")

        # 更新视频画面
        if frame is not None and frame.size > 0:
            try:
                # 获取当前实际尺寸，确保有效值
                label_w = self.video_label.width()
                label_h = self.video_label.height()

                # 如果标签尺寸无效，使用默认尺寸
                if label_w <= 0 or label_h <= 0:
                    label_w = 640
                    label_h = 480

                frame_copy = frame.copy()
                h, w = frame_copy.shape[:2]

                # 确保帧尺寸有效
                if h > 0 and w > 0:
                    cv2.putText(frame_copy, f"EAR: {ear:.3f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if ear < 0.21:
                        cv2.putText(frame_copy, "BLINKING", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                        label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    if not scaled_pixmap.isNull():
                        self.video_label.setPixmap(scaled_pixmap)
            except Exception as e:
                print(f"视频画面更新错误: {e}")


class SettingsPage(SiPage):
    """设置页面"""

    def __init__(self, app, config_manager):
        super().__init__(app)
        self.app = app
        self.config_manager = config_manager

        self.scroll_container = SiTitledWidgetGroup(self)

        # 标题
        self.title = SiLabel(self)
        self.title.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.title.setText("设置")
        self.title.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 32px; font-weight: bold;")
        self.title.setFixedHeight(50)

        # =====================================================================
        # 检测设置组
        # =====================================================================
        self.detection_group = SiOptionCardPlane(self)
        self.detection_group.setTitle("眨眼检测设置")
        # 自适应计算
        self.detection_group.setFixedHeight(300)

        # =====================================================================
        # 自适应阈值设置（放在检测设置组内）
        # =====================================================================
        # 自适应阈值开关
        self.adaptive_label = SiLabel(self.detection_group.body())
        self.adaptive_label.setText("自适应阈值")
        self.adaptive_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.adaptive_label.setFixedHeight(30)

        self.adaptive_switch = SiSwitch(self.detection_group.body())
        use_adaptive = self.config_manager.get('detection.use_adaptive_threshold', True)
        self.adaptive_switch.setChecked(use_adaptive)
        self.adaptive_switch.toggled.connect(
            lambda v: (
                self.config_manager.set('detection.use_adaptive_threshold', v),
                self.config_manager.save()
            )
        )

        self.detection_group.body().addWidget(self.adaptive_label)
        self.detection_group.body().addWidget(self.adaptive_switch)

        # 自适应窗口大小
        window_value = self.config_manager.get('detection.adaptive_window_seconds', 5.0)
        self.adaptive_window_label = SiLabel(self.detection_group.body())
        self.adaptive_window_label.setText(f"自适应窗口: {window_value:.1f} 秒")
        self.adaptive_window_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.adaptive_window_label.setFixedHeight(30)

        self.adaptive_window_slider = SiSliderH(self.detection_group.body())
        self.adaptive_window_slider.setFixedHeight(30)
        self.adaptive_window_slider.setRange(10, 100)  # 1秒到10秒
        self.adaptive_window_slider.setValue(int(window_value * 10))
        self.adaptive_window_slider.setFixedWidth(400)
        self.adaptive_window_slider.valueChanged.connect(
            lambda v: (
                self.adaptive_window_label.setText(f"自适应窗口: {v / 10:.1f} 秒"),
                self.config_manager.set('detection.adaptive_window_seconds', v / 10),
                self.config_manager.save()
            )
        )

        # 阈值比例
        ratio_value = self.config_manager.get('detection.adaptive_threshold_ratio', 0.7)
        self.adaptive_ratio_label = SiLabel(self.detection_group.body())
        self.adaptive_ratio_label.setText(f"阈值比例: {ratio_value:.2f}")
        self.adaptive_ratio_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.adaptive_ratio_label.setFixedHeight(30)

        self.adaptive_ratio_slider = SiSliderH(self.detection_group.body())
        self.adaptive_ratio_slider.setFixedHeight(30)
        self.adaptive_ratio_slider.setRange(50, 90)  # 0.50到0.90
        self.adaptive_ratio_slider.setValue(int(ratio_value * 100))
        self.adaptive_ratio_slider.setFixedWidth(400)
        self.adaptive_ratio_slider.valueChanged.connect(
            lambda v: (
                self.adaptive_ratio_label.setText(f"阈值比例: {v / 100:.2f}"),
                self.config_manager.set('detection.adaptive_threshold_ratio', v / 100),
                self.config_manager.save()
            )
        )

        self.detection_group.body().addWidget(self.adaptive_window_label)
        self.detection_group.body().addWidget(self.adaptive_window_slider)
        self.detection_group.body().addWidget(self.adaptive_ratio_label)
        self.detection_group.body().addWidget(self.adaptive_ratio_slider)

        # =====================================================================
        # MediaPipe设置组
        # =====================================================================
        self.mediapipe_group = SiOptionCardPlane(self)
        self.mediapipe_group.setTitle("面部检测设置 (MediaPipe)")
        self.mediapipe_group.setFixedHeight(280)

        # 人脸检测置信度
        det_conf_value = self.config_manager.get('mediapipe.min_detection_confidence', 0.5)
        self.det_conf_label = SiLabel(self.mediapipe_group.body())
        self.det_conf_label.setText(f"人脸检测置信度: {det_conf_value:.1f}")
        self.det_conf_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.det_conf_label.setFixedHeight(30)

        self.det_conf_slider = SiSliderH(self.mediapipe_group.body())
        self.det_conf_slider.setFixedHeight(30)
        self.det_conf_slider.setRange(3, 8)
        self.det_conf_slider.setValue(int(det_conf_value * 10))
        self.det_conf_slider.setFixedWidth(400)
        self.det_conf_slider.valueChanged.connect(
            lambda v: (
                self.det_conf_label.setText(f"人脸检测置信度: {v / 10:.1f}"),
                self.config_manager.set('mediapipe.min_detection_confidence', v / 10),
                self.config_manager.save()
            )
        )

        # 关键点跟踪置信度
        track_conf_value = self.config_manager.get('mediapipe.min_tracking_confidence', 0.3)
        self.track_conf_label = SiLabel(self.mediapipe_group.body())
        self.track_conf_label.setText(f"关键点跟踪置信度: {track_conf_value:.1f}")
        self.track_conf_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.track_conf_label.setFixedHeight(30)

        self.track_conf_slider = SiSliderH(self.mediapipe_group.body())
        self.track_conf_slider.setFixedHeight(30)
        self.track_conf_slider.setRange(2, 7)
        self.track_conf_slider.setValue(int(track_conf_value * 10))
        self.track_conf_slider.setFixedWidth(400)
        self.track_conf_slider.valueChanged.connect(
            lambda v: (
                self.track_conf_label.setText(f"关键点跟踪置信度: {v / 10:.1f}"),
                self.config_manager.set('mediapipe.min_tracking_confidence', v / 10),
                self.config_manager.save()
            )
        )

        self.mediapipe_group.body().addWidget(self.det_conf_label)
        self.mediapipe_group.body().addWidget(self.det_conf_slider)
        self.mediapipe_group.body().addWidget(self.track_conf_label)
        self.mediapipe_group.body().addWidget(self.track_conf_slider)

        # =====================================================================
        # 提醒设置组
        # =====================================================================
        self.alert_group = SiOptionCardPlane(self)
        self.alert_group.setTitle("提醒设置")
        self.alert_group.setFixedHeight(320)

        # 启用提醒开关
        self.notification_switch = SiSwitch(self.alert_group.body())
        self.notification_switch.setChecked(self.config_manager.get('alert.enabled', True))
        self.notification_switch.setFixedSize(60, 30)
        self.notification_switch.toggled.connect(
            lambda state: (
                self.config_manager.set('alert.enabled', state),
                self.config_manager.save()
            )
        )

        self.notification_label = SiLabel(self.alert_group.body())
        self.notification_label.setText("启用系统通知")
        self.notification_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.notification_label.setFixedHeight(30)

        self.alert_group.body().addWidget(self.notification_label)
        self.alert_group.body().addWidget(self.notification_switch)

        # 语音播报开关
        self.voice_switch = SiSwitch(self.alert_group.body())
        self.voice_switch.setChecked(self.config_manager.get('alert.voice_enabled', False))
        self.voice_switch.setFixedSize(60, 30)

        self.voice_label = SiLabel(self.alert_group.body())
        self.voice_label.setText("启用语音播报")
        self.voice_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.voice_label.setFixedHeight(30)

        self.voice_switch.toggled.connect(
            lambda state: (
                self.config_manager.set('alert.voice_enabled', state),
                self.config_manager.save()
            )
        )

        self.alert_group.body().addWidget(self.voice_label)
        self.alert_group.body().addWidget(self.voice_switch)

        # 提醒间隔
        alert_interval_value = self.config_manager.get('alert.alert_interval', 30)
        self.alert_interval_label = SiLabel(self.alert_group.body())
        self.alert_interval_label.setText(f"提醒间隔: {alert_interval_value}秒")
        self.alert_interval_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.alert_interval_label.setFixedHeight(30)

        self.alert_interval_slider = SiSliderH(self.alert_group.body())
        self.alert_interval_slider.setFixedHeight(30)
        self.alert_interval_slider.setRange(10, 300)  # 10秒到5分钟
        self.alert_interval_slider.setValue(alert_interval_value)
        self.alert_interval_slider.setFixedWidth(400)
        self.alert_interval_slider.valueChanged.connect(
            lambda v: (
                self.alert_interval_label.setText(f"提醒间隔: {v}秒"),
                self.config_manager.set('alert.alert_interval', v),
                self.config_manager.save()
            )
        )

        self.alert_group.body().addWidget(self.alert_interval_label)
        self.alert_group.body().addWidget(self.alert_interval_slider)

        # =====================================================================
        # 摄像头设置组
        # =====================================================================
        self.camera_group = SiOptionCardPlane(self)
        self.camera_group.setTitle("摄像头设置")
        self.camera_group.setFixedHeight(250)

        # 摄像头索引
        camera_index_value = self.config_manager.get('camera.camera_index', 0)
        self.camera_index_label = SiLabel(self.camera_group.body())
        self.camera_index_label.setText(f"摄像头索引: {camera_index_value}")
        self.camera_index_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.camera_index_label.setFixedHeight(30)

        self.camera_index_slider = SiSliderH(self.camera_group.body())
        self.camera_index_slider.setFixedHeight(30)
        self.camera_index_slider.setRange(0, 5)
        self.camera_index_slider.setValue(camera_index_value)
        self.camera_index_slider.setFixedWidth(400)
        self.camera_index_slider.valueChanged.connect(
            lambda v: (
                self.camera_index_label.setText(f"摄像头索引: {v}"),
                self.config_manager.set('camera.camera_index', v),
                self.config_manager.save()
            )
        )

        # 目标帧率
        target_fps_value = self.config_manager.get('camera.target_fps', 50)
        self.camera_fps_label = SiLabel(self.camera_group.body())
        self.camera_fps_label.setText(f"目标帧率: {int(target_fps_value)} FPS")
        self.camera_fps_label.setStyleSheet(f"color: {get_color(SiColor.TEXT_A)}; font-size: 16px;")
        self.camera_fps_label.setFixedHeight(30)

        self.camera_fps_slider = SiSliderH(self.camera_group.body())
        self.camera_fps_slider.setFixedHeight(30)
        self.camera_fps_slider.setRange(10, 60)
        self.camera_fps_slider.setValue(int(target_fps_value))
        self.camera_fps_slider.setFixedWidth(400)
        self.camera_fps_slider.valueChanged.connect(
            lambda v: (
                self.camera_fps_label.setText(f"目标帧率: {v} FPS"),
                self.config_manager.set('camera.target_fps', float(v)),
                self.config_manager.save()
            )
        )

        self.camera_group.body().addWidget(self.camera_index_label)
        self.camera_group.body().addWidget(self.camera_index_slider)
        self.camera_group.body().addWidget(self.camera_fps_label)
        self.camera_group.body().addWidget(self.camera_fps_slider)

        # =====================================================================
        # 添加所有组到滚动容器
        # =====================================================================
        self.scroll_container.addWidget(self.title)
        self.scroll_container.addWidget(self.detection_group)
        self.scroll_container.addWidget(self.mediapipe_group)
        # self.scroll_container.addWidget(self.perclos_group)
        self.scroll_container.addWidget(self.alert_group)
        self.scroll_container.addWidget(self.camera_group)

        self.setAttachment(self.scroll_container)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = event.size().width()
        self.scroll_container.setFixedWidth(min(w - 128, 800))


class MySiliconApp(SiliconApplication):
    """主应用类"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screen_geo = QDesktopWidget().screenGeometry()
        # self.setMinimumSize(1000, 600)
        self.resize(1000, 700)
        self.move((screen_geo.width() - self.width()) // 2,
                  (screen_geo.height() - self.height()) // 2)

        self.layerMain().setTitle("给我眨眼睛 - 干眼检测器")
        self.setWindowTitle("给我眨眼睛 - 干眼检测器")

        # 设置窗口图标 (将图标文件放在项目根目录的 img 文件夹中)
        # 例如: img/app_icon.png 或 img/app_icon.ico
        import os
        icon_path = os.path.join(os.path.dirname(__file__), 'img', 'app_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        # 初始化管理器
        self.camera_manager = CameraManager()
        # self.camera_manager = camera_manager
        self.config_manager = ConfigManager()

        # 状态变量
        self.is_detecting = False
        self.detection_thread = None
        self.alert_thread = None  # 提醒线程
        self.eye_fatigue_analyzer = EyeFatigueAnalyzer(self.config_manager)
        self.current_frame = None
        self.current_ear = 0.3
        self.current_blink_rate = 15.0
        self.blink_count = 0

        # 创建页面
        self.monitor_page = MonitorPage(self)
        self.settings_page = SettingsPage(self, self.config_manager)

        # 添加页面到导航栏
        self.layerMain().addPage(
            self.monitor_page,
            icon=SiGlobal.siui.iconpack.get("ic_fluent_eye_filled"),
            hint="监控",
            side="top"
        )
        self.layerMain().addPage(
            self.settings_page,
            icon=SiGlobal.siui.iconpack.get("ic_fluent_settings_filled"),
            hint="设置",
            side="top"
        )

        self.layerMain().setPage(0)

        # 连接信号
        self.monitor_page.start_button.clicked.connect(self.toggle_detection)
        self.monitor_page.reset_button.clicked.connect(self.reset_statistics)

        # 定时更新UI
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)
        
        # 重新加载样式
        SiGlobal.siui.reloadAllWindowsStyleSheet()

    def toggle_detection(self):
        """切换检测状态"""
        if self.is_detecting:
            self.stop_detection()
        else:
            self.start_detection()

    def start_detection(self):
        """开始检测"""
        if self.is_detecting:
            return

        self.is_detecting = True
        self.monitor_page.start_button.attachment().setText("停止检测")

        # 启动检测线程
        self.detection_thread = BlinkDetectionThread(self.camera_manager, self.config_manager)
        self.detection_thread.status_update.connect(self.on_status_update)
        # self.detection_thread.blink_detected.connect(self.on_blink_detected)
        self.detection_thread.start()

        # self.camera_manager.frame_provider.start()

        # 启动提醒线程
        self.alert_thread = AlertThread(self.config_manager, self.eye_fatigue_analyzer)
        # self.alert_thread.alert_triggered.connect(self.on_alert_triggered)
        # self.alert_thread.notification_failed.connect(self.on_notification_failed)
        
        # 从配置读取提醒设置
        use_voice = self.config_manager.get('alert.voice_enabled', False)
        use_notification = self.config_manager.get('alert.enabled', True)
        alert_interval = self.config_manager.get('alert.alert_interval', 30)
        self.alert_thread.update_config(use_voice, use_notification, alert_interval)
        self.alert_thread.start()

    def stop_detection(self):
        """停止检测"""
        self.is_detecting = False
        self.monitor_page.start_button.attachment().setText("开始检测")

        # 停止检测线程
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread = None

        # 停止提醒线程
        if self.alert_thread:
            self.alert_thread.stop()
            self.alert_thread = None

    def reset_statistics(self):
        """重置统计"""
        self.blink_count = 0
        self.eye_fatigue_analyzer.reset()

        if self.detection_thread:
            self.detection_thread.blink_detector.reset()

    def on_status_update(self, status):
        """状态更新回调"""
        # self.current_frame = status.get('frame')
        # 必须使用 .copy()，否则主线程更新 UI 时如果子线程修改了内存，程序会崩溃
        raw_frame = status.get('frame')
        if raw_frame is not None:
            self.current_frame = raw_frame.copy()
        
        self.current_ear = status.get('ear', 0.3)
        self.blink_count = status.get('statistics', {}).get('current_blink_count', 0)
        self.current_blink_rate = status.get('statistics', {}).get('blink_rate_per_minute', 0.0)
        self.eye_fatigue_analyzer.update(self.current_ear)
        # 同步频率：BlinkDetector 算的是 60s 窗口，与 UI 数字完全对齐
        self.eye_fatigue_analyzer.set_blink_rate(self.current_blink_rate)

    # def on_blink_detected(self, blink_info):
    #     """眨眼检测回调"""
    #     # 记录眨眼事件到疲劳分析器（用于计算眨眼频率）
    #     # self.eye_fatigue_analyzer.record_blink()
    #     blink_rate = self.detection_thread.blink_detector.get_blink_rate()
    #     self.eye_fatigue_analyzer.set_blink_rate(blink_rate)


    # def on_alert_triggered(self, message: str):
    #     """提醒触发回调（可选：可在此添加UI反馈）"""
    #     print(f"提醒触发: {message}")

    # def on_notification_failed(self, message: str):
    #     """通知失败回调，使用消息框作为备选"""
    #     from PyQt5.QtWidgets import QMessageBox
    #     QMessageBox.warning(self, "眨眼护眼提醒", message)

    def update_ui(self):
        """更新UI"""
        perclos = 0.2
        analysis = self.eye_fatigue_analyzer.analyze_fatigue()
        health_level = analysis['health_level']
        health_score = analysis['health_score']

        self.monitor_page.update_stats(
            self.blink_count,
            self.current_blink_rate,
            perclos,
            self.current_ear,
            health_level,
            health_score,
            self.current_frame
        )

    def closeEvent(self, event):
        """关闭窗口事件"""
        if self.is_detecting:
            self.stop_detection()
        self.camera_manager.cleanup()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MySiliconApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
