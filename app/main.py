"""DataCopilot Mini —— Streamlit 入口：三页导航。

页面划分（SPEC 契约）：

1. 📊 数据分析：上传 / 示例 CSV → 自动数据剖析 → 自然语言查询（NL2SQL → DuckDB → 图）
2. 📚 知识库：选文档入库 → RAG 问答（带引用来源）
3. 🧩 低代码看板：左侧 YAML 编辑器 → 右侧即时渲染看板

所有页面代码都在函数里，模块级别只做导入与常量定义，因此无头环境
（``python -c "import app.main"``）也能安全导入。
"""

from __future__ import annotations

import sys
from pathlib import Path

# streamlit run app/main.py 时脚本目录会被加进 sys.path，这里补上项目根，
# 保证 `from app.xxx import ...` 在任意工作目录下都能解析。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.analysis.nl2sql import ask_nl2sql  # noqa: E402
from app.analysis.profile import profile_csv  # noqa: E402
from app.config import ConfigError, PipelineConfig  # noqa: E402
from app.kb.rag import RAG  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.lowcode.dashboard import render_dashboard  # noqa: E402
from app.lowcode.pipeline import PipelineError, run_pipeline  # noqa: E402

try:  # pragma: no cover - 取决于本机是否装了 streamlit（缺了也能 import 本模块）
    import streamlit as st
except ImportError as _exc:  # pragma: no cover
    st = None
    _ST_IMPORT_ERROR: ImportError | None = _exc
else:
    _ST_IMPORT_ERROR = None

# ---- 路径常量 ----
SAMPLE_CSV = ROOT / "data" / "sales.csv"
KB_DIR = ROOT / "data" / "kb"
CONFIG_DIR = ROOT / "configs"
UPLOAD_DIR = ROOT / ".cache" / "uploads"

#: 「新建配置」按钮写入编辑器的模板（低代码上手用的最小可用配置）
TEMPLATE_YAML = """# 新建看板：只改这份 YAML 就能生成一整页，无需动 Python
name: 我的看板
data: data/sales.csv
transforms:
  - id: by_region
    op: aggregate
    group_by: [region]
    metrics:
      total_amount: {col: amount, fn: sum}
charts:
  - type: bar
    data: by_region
    x: region
    y: total_amount
    title: 区域销售额
filters:
  - field: category
    kind: multiselect
"""


def _st():
    """取 streamlit 模块；未安装时抛出安装提示。"""
    if st is None:  # pragma: no cover - 仅在缺依赖时触发
        raise RuntimeError("未安装 streamlit，请先执行 pip install -r requirements.txt") from _ST_IMPORT_ERROR
    return st


# ------------------------------------------------------------------ 公共小工具


def _client() -> LLMClient:
    """会话级复用同一个 LLM 客户端（模式徽标与各页共享）。"""
    st = _st()
    if "llm_client" not in st.session_state:
        st.session_state["llm_client"] = LLMClient()
    return st.session_state["llm_client"]


def _mode_badge(client: LLMClient) -> str:
    """当前 LLM 模式文案。"""
    return "mock（内置规则，零网络）" if client.mock else "dashscope（真实 API）"


def _materialize(df: pd.DataFrame, name: str) -> Path:
    """把上传的 DataFrame 落盘，便于复用 profile_csv（它按路径读文件）。"""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    df.to_csv(path, index=False)
    return path


def _auto_chart(df: pd.DataFrame, *, max_rows: int = 20):
    """给查询结果随便配一张能看的图：第一个文本列做 x，第一个数值列做 y。"""
    import plotly.express as px

    if df.empty:
        return None
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not text_cols or not num_cols:
        return None
    x, y = text_cols[0], num_cols[0]
    data = df.head(max_rows)
    fig = px.bar(data, x=x, y=y, title=f"{y} by {x}")
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=340)
    return fig


# ------------------------------------------------------------------ 页 1：数据分析


def page_analysis() -> None:
    st = _st()
    st.header("📊 数据分析")
    st.caption("上传或使用示例 CSV → 自动数据剖析 → 用中文直接查数据（NL2SQL → DuckDB → 图）")

    source = st.radio("数据来源", ["示例数据 data/sales.csv", "上传 CSV"], horizontal=True)
    df: pd.DataFrame | None = None
    csv_path: Path | None = None

    if source.startswith("示例"):
        if SAMPLE_CSV.exists():
            df, csv_path = pd.read_csv(SAMPLE_CSV), SAMPLE_CSV
        else:
            st.error(f"示例数据不存在：{SAMPLE_CSV}")
    else:
        uploaded = st.file_uploader("选择 CSV 文件", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            csv_path = _materialize(df, "uploaded.csv")

    if df is None or csv_path is None:
        st.info("先选一份数据，页面会自动给出剖析结果。")
        return

    overview_tab, query_tab = st.tabs(["数据概览", "自然语言查询"])

    # ---- 概览：直接渲染 profile 的结构化输出 ----
    with overview_tab:
        prof = profile_csv(csv_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("行数", prof["shape"]["rows"])
        c2.metric("列数", prof["shape"]["columns"])
        c3.metric("重复行率", f"{prof['duplicates']['rate']:.1%}")
        c4.metric("缺失率最高", f"{max(prof['missing'].values(), default=0):.1%}")

        st.subheader("列信息")
        st.dataframe(
            pd.DataFrame(
                {
                    "列名": prof["columns"],
                    "类型": [prof["dtypes"][c] for c in prof["columns"]],
                    "缺失率": [f"{prof['missing'][c]:.1%}" for c in prof["columns"]],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        if prof["numeric"]["describe"]:
            st.subheader("数值列描述统计")
            st.dataframe(
                pd.DataFrame(prof["numeric"]["describe"]).T, use_container_width=True
            )

        if prof["top_values"]:
            st.subheader("类别列 Top 分布")
            for col, counts in prof["top_values"].items():
                with st.expander(f"{col}（{len(counts)} 个取值）"):
                    st.bar_chart(pd.Series(counts, name="行数"))

    # ---- 查询：NL2SQL ----
    with query_tab:
        st.caption("例：各地区销售额是多少？ / 销量最高的品类是什么？")
        question = st.text_input("用中文提问", key="nl2sql_question")
        if st.button("查询", type="primary") and question.strip():
            result = ask_nl2sql(question.strip(), df=df, client=_client())
            st.session_state["nl2sql_result"] = result

        result = st.session_state.get("nl2sql_result")
        if result is not None:
            if result.success:
                st.code(result.sql or "", language="sql")
                st.caption(f"尝试 {result.attempts} 次 · 返回 {len(result.df)} 行")
                st.dataframe(result.df, use_container_width=True, hide_index=True)
                fig = _auto_chart(result.df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"SQL 执行失败（已重试 {result.attempts - 1} 次）：{result.error}")
                if result.sql:
                    st.code(result.sql, language="sql")


# ------------------------------------------------------------------ 页 2：知识库


def _kb_documents() -> list[str]:
    """收集入库文档：内置 Markdown + 上传的 Markdown（落到 .cache/uploads）。"""
    st = _st()
    builtin = sorted(KB_DIR.glob("*.md"))
    names = st.multiselect(
        "内置知识文档",
        options=[p.name for p in builtin],
        default=[p.name for p in builtin],
    )
    docs = [str(KB_DIR / n) for n in names]

    uploads = st.file_uploader(
        "追加 Markdown / 文本文件", type=["md", "markdown", "txt"], accept_multiple_files=True
    )
    for item in uploads or []:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / Path(item.name).name
        path.write_text(item.getvalue().decode("utf-8", errors="replace"), encoding="utf-8")
        docs.append(str(path))
    return docs


def page_kb() -> None:
    st = _st()
    st.header("📚 知识库")
    st.caption("文档切块 → 向量化 → 检索增强问答，回答附带引用来源")

    docs = _kb_documents()
    if not docs:
        st.info("至少选择或上传一篇文档。")
        return

    signature = tuple(docs)
    if st.button("构建 / 更新向量库", type="primary") or st.session_state.get("kb_signature") != signature:
        try:
            rag = RAG(client=_client())
            rag.build(list(signature))
            st.session_state["kb_rag"] = rag
            st.session_state["kb_signature"] = signature
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"入库失败：{exc}")
            return

    rag = st.session_state.get("kb_rag")
    if rag is None or rag.store is None:
        return
    st.success(f"已索引 {len(rag.store.meta['texts'])} 个文本块 · {len(signature)} 篇文档")

    question = st.text_input("向知识库提问", key="kb_question", placeholder="例如：退货期限是多久？")
    if question.strip() and st.button("提问"):
        st.session_state["kb_result"] = rag.answer(question.strip())

    result = st.session_state.get("kb_result")
    if result:
        st.markdown("**回答**")
        st.write(result["answer"])
        if result["sources"]:
            st.markdown("**参考来源**")
            for src in result["sources"]:
                st.markdown(
                    f"- `{Path(src['source']).name}` · {src['heading'] or '（无标题）'}"
                    f" · 相似度 {src['similarity']}"
                )


# ------------------------------------------------------------------ 页 3：低代码看板


def _parse_config_text(text: str) -> tuple[PipelineConfig | None, str | None]:
    """把编辑器里的 YAML 文本校验成 PipelineConfig，返回 (配置, 错误信息)。"""
    try:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            return None, "顶层必须是映射（key: value）"
        return PipelineConfig.model_validate(raw), None
    except yaml.YAMLError as exc:
        return None, f"YAML 语法错误：{exc}"
    except ValidationError as exc:
        return None, f"配置校验失败：{exc}"
    except ConfigError as exc:
        return None, str(exc)


def page_lowcode() -> None:
    st = _st()
    st.header("🧩 低代码看板")
    st.caption("左边改 YAML，右边立刻变看板 —— 页面上的每张图、每个筛选器都由配置驱动")

    names = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
    left, right = st.columns([1, 2], gap="large")

    with left:
        picked = st.selectbox("配置文件", options=names or ["（无）"])
        if st.button("新建配置（模板）"):
            st.session_state["cfg_yaml"] = TEMPLATE_YAML
            # 标记"已同步"，避免下面的回填逻辑把刚载入的模板覆盖掉
            st.session_state["_cfg_picked"] = picked
        # 切换配置文件时把磁盘内容读进编辑器
        if names and st.session_state.get("_cfg_picked") != picked:
            st.session_state["cfg_yaml"] = (CONFIG_DIR / picked).read_text(encoding="utf-8")
            st.session_state["_cfg_picked"] = picked
        st.session_state.setdefault("cfg_yaml", TEMPLATE_YAML)

        text = st.text_area("YAML 配置", key="cfg_yaml", height=380)

        save_name = st.text_input("保存为（configs/ 下）", value=picked if names else "my_dashboard.yaml")
        if st.button("保存配置"):
            target = CONFIG_DIR / Path(save_name).name
            target.write_text(text, encoding="utf-8")
            st.success(f"已保存：{target.relative_to(ROOT)}")

    cfg, error = _parse_config_text(text)
    with right:
        if error or cfg is None:
            st.error(error or "配置为空")
            return
        st.subheader(cfg.name)
        data_path = Path(cfg.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        if not data_path.exists():
            st.warning(f"数据文件不存在：{data_path}")
            return
        try:
            df = pd.read_csv(data_path)
            datasets = run_pipeline(df, cfg.transforms)
            render_dashboard(cfg, datasets)
        except (PipelineError, KeyError, ValueError) as exc:
            st.error(f"渲染失败：{exc}")


# ------------------------------------------------------------------ 入口


def main() -> None:
    """设置页面、顶栏模式徽标，并挂上三个页面。"""
    st = _st()
    st.set_page_config(page_title="DataCopilot Mini", page_icon="🧭", layout="wide")

    client = _client()
    st.markdown(
        f"### 🧭 DataCopilot Mini &nbsp;·&nbsp; LLM 模式：`{'mock' if client.mock else 'dashscope'}`"
    )
    st.caption(_mode_badge(client))

    pages = [
        st.Page(page_analysis, title="数据分析", icon="📊", url_path="analysis", default=True),
        st.Page(page_kb, title="知识库", icon="📚", url_path="knowledge"),
        st.Page(page_lowcode, title="低代码看板", icon="🧩", url_path="lowcode"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()