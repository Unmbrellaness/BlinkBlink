"""
提醒模块 - 独立工作线程
处理音频播放和系统通知，避免阻塞UI线程
"""
import time
import threading
import os
from PyQt5.QtCore import QThread, pyqtSignal


class AlertThread(QThread):
    """提醒线程 - 独立处理音频播放和系统通知"""
    alert_triggered = pyqtSignal(str)  # 提醒触发信号
    notification_failed = pyqtSignal(str)  # 通知失败信号（回退到主线程）

    def __init__(self, config_manager, fatigue_analyzer):
        super().__init__()
        self.config_manager = config_manager
        self.fatigue_analyzer = fatigue_analyzer
        self.running = False
        self.lock = threading.Lock()
        
        # 配置参数
        self._use_voice = False
        self._use_notification = True
        self._alert_interval = 30
        
    def update_config(self, use_voice: bool, use_notification: bool, alert_interval: int):
        """更新提醒配置"""
        with self.lock:
            self._use_voice = use_voice
            self._use_notification = use_notification
            self._alert_interval = alert_interval

    def run(self):
        """提醒线程主循环"""
        self.running = True
        last_alert_time = 0
        
        while self.running:
            current_time = time.time()
            
            # 读取当前配置
            with self.lock:
                interval = self._alert_interval
                use_voice = self._use_voice
                use_notification = self._use_notification
            
            # 检查是否到提醒时间
            if current_time - last_alert_time >= interval:
                analysis = self.fatigue_analyzer.analyze_fatigue()
                health_level = analysis['health_level']
                
                # 只在需要提醒的状态下触发
                if health_level in ["提醒", "需注意"]:
                    recommendation = self.fatigue_analyzer.get_recommendation()
                    if recommendation:
                        # 发送信号通知UI（可做额外UI反馈）
                        self.alert_triggered.emit(recommendation)
                        
                        # 播放音频（在新线程中播放，不阻塞）
                        if use_voice:
                            self._play_audio_async()
                        
                        # 系统通知
                        if use_notification:
                            self._send_notification(recommendation)
                        
                        last_alert_time = current_time
            
            time.sleep(1)  # 每秒检查一次

    def _play_audio_async(self):
        """异步播放音频（在独立线程中播放）"""
        def _play():
            try:
                # 延迟导入pygame，避免启动时卡顿
                import pygame
                
                mp3_path = os.path.join(os.path.dirname(__file__), '放松下眼睛吧.mp3')
                if os.path.exists(mp3_path):
                    pygame.mixer.init()
                    pygame.mixer.music.load(mp3_path)
                    pygame.mixer.music.play()
                    print(f"播放音频提醒: {mp3_path}")
                else:
                    print(f"音频文件不存在: {mp3_path}")
            except Exception as e:
                print(f"播放音频时出错: {e}")
        
        threading.Thread(target=_play, daemon=True).start()

    def _send_notification(self, message: str):
        """发送系统通知"""
        try:
            from plyer import notification
            notification.notify(
                title="眨眼护眼提醒",
                message=message,
                app_name="给我眨眼睛",
                # app_icon=os.path.join(os.path.dirname(__file__), 'icon.png'),
                timeout=20
            )
        except Exception as e:
            print(f"显示通知时出错: {e}")
            # 通知主线程用备选方案（消息框）
            self.notification_failed.emit(message)

    def stop(self):
        """停止提醒线程"""
        self.running = False
        self.wait()
