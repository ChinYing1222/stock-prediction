"""pytest 設定：把專案根目錄加入匯入路徑，讓測試可以 import config 與 src。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
