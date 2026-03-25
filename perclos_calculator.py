"""
PERCLOS计算器 - 眼睛闭合百分比计算
PERCLOS (Percentage of Eye Closure) 是一种衡量眼睛疲劳程度的指标
"""
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque

class EyeFatigueAnalyzer:
    """眼睛护眼分析器 - 基于眨眼频率分析眼睛护眼状态"""

    def __init__(self, config_manager=None):
        """初始化眼睛护眼分析器"""
        self.ear_history = deque(maxlen=300)  # 保留，仅用于 avg_ear 显示
        self._blink_rate = 0.0  # 由外部 BlinkDetector 注入，不再自己算

        # 护眼阈值
        self.health_thresholds = {
            'blink_rate_excellent': 25.0,  # 优秀：>=25次/分钟
            'blink_rate_good': 15.0,       # 良好：>=15次/分钟
            'blink_rate_fair': 8.0,        # 一般：>=8次/分钟
            'blink_rate_warning': 5.0,     # 需注意：>=5次/分钟
        }
        
    def set_blink_rate(self, blink_rate: float):
        """由外部 BlinkDetector 注入当前眨眼频率（次/分钟）"""
        self._blink_rate = blink_rate
    
    def update(self, ear: float, timestamp: float = None):
        """
        更新分析数据

        Args:
            ear: 当前EAR值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        # 更新EAR历史
        self.ear_history.append({
            'ear': ear,
            'timestamp': timestamp
        })

    def record_blink(self, timestamp: float = None):
        """
        记录一次眨眼事件，用于计算眨眼频率

        Args:
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
        self.blink_history.append({
            'timestamp': timestamp
        })

    def analyze_fatigue(self) -> Dict:
        """使用外部注入的眨眼频率，不再自己计算"""
        # 直接用外部注入的频率，不再查 blink_history
        blink_rate = self._blink_rate
        avg_ear = sum(d['ear'] for d in self.ear_history) / len(self.ear_history) \
                  if self.ear_history else 0.3
        thresholds = self.health_thresholds
        if blink_rate >= thresholds['blink_rate_excellent']:
            health_score = 100
            health_level = "优秀"
        elif blink_rate >= thresholds['blink_rate_good']:
            health_score = int(75 + (blink_rate - thresholds['blink_rate_good']) /
                              (thresholds['blink_rate_excellent'] - thresholds['blink_rate_good']) * 24)
            health_level = "良好"
        elif blink_rate >= thresholds['blink_rate_fair']:
            health_score = int(50 + (blink_rate - thresholds['blink_rate_fair']) /
                              (thresholds['blink_rate_good'] - thresholds['blink_rate_fair']) * 25)
            health_level = "一般"
        elif blink_rate >= thresholds['blink_rate_warning']:
            health_score = int(25 + (blink_rate - thresholds['blink_rate_warning']) /
                              (thresholds['blink_rate_fair'] - thresholds['blink_rate_warning']) * 25)
            health_level = "需注意"
        else:
            health_score = int(blink_rate / thresholds['blink_rate_warning'] * 25)
            health_level = "提醒"
        health_reasons = []
        if blink_rate < thresholds['blink_rate_fair']:
            health_reasons.append(f"眨眼频率偏低 ({blink_rate:.1f}次/分钟)，建议多眨眼保护眼睛")
        return {
            'health_score': max(0, min(100, health_score)),
            'health_level': health_level,
            'reasons': health_reasons,
            'perclos': 0.2,
            'blink_rate': blink_rate,       # 与 UI 显示的眨眼频率完全一致
            'avg_ear': avg_ear,
            'total_ear_records': len(self.ear_history),
            'total_blink_records': 0,       # 不再使用 blink_history
            'thresholds': self.health_thresholds
        }

    def get_recommendation(self) -> Optional[str]:
        """护眼建议，直接从当前频率推"""
        analysis = self.analyze_fatigue()
        level = analysis['health_level']
        recommendations = {
            "提醒": "⚠️ 眨眼频率过低！请立即眨眼或闭眼休息5-10秒，保护眼睛。",
            "需注意": "👁️ 眨眼频率偏低，建议多眨眼，保持眼睛湿润。",
            "一般": "💡 眨眼频率一般，建议有意识地多眨眼。",
            "良好": "👍 眨眼频率良好，继续保持！",
            "优秀": "🌟 眨眼频率非常棒！护眼状态优秀！",
        }
        return recommendations.get(level)

    def reset(self):
        """重置分析器"""
        self.ear_history.clear()
        self._blink_rate = 0.0  # 重置频率
