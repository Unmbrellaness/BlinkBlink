import sys
import os

# 添加PyQt-SiliconUI到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PyQt-SiliconUI-main'))

# 导入必要的模块
import siui
from siui.core import SiColor, SiGlobal, Si, GlobalFont, GlobalFontSize
from siui.core.globals import SiGlobal
from siui.templates.application.application import SiliconApplication
from siui.components.page import SiPage
from siui.components.label import SiLabel
from siui.components.slider import SiSliderH
from siui.components.titled_widget_group import SiTitledWidgetGroup
from siui.components.option_card import SiOptionCardPlane
from siui.components.widgets import (
    SiDenseHContainer,
    SiDenseVContainer,
    SiPushButton,
    SiSwitch,
)
from siui.components.progress_bar import SiProgressBar
from siui.gui import SiFont

# 导入本地模块
from camera_manager import CameraManager
from eye_blink_detector import EARCalculator, BlinkDetector
from perclos_calculator import EyeFatigueAnalyzer
from config_manager import ConfigManager

# 图标路径
ICONS_DIR = os.path.join(os.path.dirname(__file__), 'PyQt-SiliconUI-main', 'examples', 'Gallery for siui', 'icons')
