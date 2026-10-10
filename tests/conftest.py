"""pytest 公共配置。

做两件事：
1. 把项目根目录加入 sys.path，这样直接运行 `pytest -q`（不依赖 `python -m pytest`）
   也能 import app/eval 包。
2. 强制测试会话进入 mock 模式，并清掉机器上可能存在的真实 API Key——
   SPEC 要求单测不依赖网络、不写死 Key，这里从环境层面兜住。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 测试永远走 mock：不触网、不消耗额度、结果可复现
os.environ["USE_MOCK"] = "1"
os.environ.pop("DASHSCOPE_API_KEY", None)