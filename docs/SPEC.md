# DataCopilot Mini — 规格书（SPEC）

一个面向 FDE 简历的「低代码 + 知识库 + 数据分析 + AI Harness」一体化小项目。
技术栈：Python 3.13 + Streamlit + pandas + DuckDB + DashScope（通义千问 qwen-plus / text-embedding-v3），必须支持无 API Key 的 mock 模式。

## 定位

上传一份 CSV + 几篇 Markdown 文档，就能：
1. 用自然语言查询数据（NL2SQL → DuckDB 执行 → Plotly 图）
2. 对文档做 RAG 问答（带引用来源）
3. 用一份 YAML「低代码」配置自动生成数据看板页
4. 用内置 AI Harness 对以上 AI 能力跑回归评测，输出 JSON/Markdown 报告

## 目录结构（严格遵守）

```
fde-data-copilot/
  README.md
  requirements.txt
  .env.example
  app/
    __init__.py
    main.py                  # Streamlit 入口（三页导航）
    config.py                # YAML 配置加载 + schema 校验
    llm/
      __init__.py
      client.py              # DashScope 客户端 + mock 模式
    lowcode/
      __init__.py
      pipeline.py            # 配置驱动的 transform 管道
      dashboard.py           # 配置 → Streamlit 看板页渲染
    kb/
      __init__.py
      chunker.py             # 文档分块
      vectorstore.py         # numpy 向量库（npz 持久化）
      rag.py                 # 检索 + 生成 + 引用
    analysis/
      __init__.py
      profile.py             # CSV 自动数据剖析
      nl2sql.py              # 自然语言 → SQL → DuckDB 执行
  eval/
    __init__.py
    harness.py               # 评测运行器（CLI: python -m eval.harness run）
    cases/
      nl2sql.jsonl           # 黄金用例
      rag.jsonl
  configs/
    sales_dashboard.yaml     # 低代码看板示例配置
  data/
    sales.csv                # 示例数据（>=500 行，列：date,region,category,salesperson,amount,quantity）
    kb/                      # >=3 篇中文 Markdown 知识文档（示例：产品FAQ、退货政策、销售流程）
  tests/
    test_config.py
    test_pipeline.py
    test_chunker.py
    test_vectorstore.py
    test_profile.py
    test_harness.py
```

## 模块契约

### app/llm/client.py

```python
class LLMClient:
    def __init__(self, api_key: str | None = None):  # 缺省读 env DASHSCOPE_API_KEY
        # 无 key 或 USE_MOCK=1 -> mock 模式
    def chat(self, prompt: str, system: str = "", model: str = "qwen-plus") -> str: ...
    def embed(self, texts: list[str], model: str = "text-embedding-v3") -> list[list[float]]: ...
    @property
    def mock(self) -> bool: ...
```

- 真实模式用 `dashscope` SDK（`dashscope.Generation.call` / `dashscope.TextEmbedding.call`）。
- mock 模式：`chat` 返回可预测的规则化回答（见下），`embed` 返回基于文本哈希的 128 维伪向量（同文本同向量，保证检索可复现）。
- mock chat 规则（保证无 key 也能演示 + 评测跑通）：
  - prompt 中含 "SQL" 关键字 → 从问题里提取列名，返回 `SELECT ... FROM sales LIMIT 5;` 形式的合法 SQL（包在 ```sql 代码块中）。
  - 其他 → 返回 `"[mock] " + prompt[:80]`。
- 所有 LLM 调用写入 `logs/traces.jsonl`：时间戳、模型、耗时、prompt 前 200 字、响应前 200 字、是否 mock。

### app/config.py

- `load_pipeline(path) -> PipelineConfig`；用 pydantic 校验。
- PipelineConfig schema:

```yaml
# configs/sales_dashboard.yaml
name: 销售驾驶舱
data: data/sales.csv
transforms:            # 按序执行，每步输出新命名数据集
  - id: monthly
    op: aggregate
    group_by: [month]  # month 由 date 派生（strftime %Y-%m）
    metrics: {total_amount: {col: amount, fn: sum}, avg_amount: {col: amount, fn: mean}}
charts:
  - type: line         # line|bar|pie
    data: monthly
    x: month
    y: total_amount
    title: 月度销售额
  - type: bar
    data: raw
    group_by: region
    metric: {col: amount, fn: sum}
    title: 区域销售额
filters:               # 全局筛选控件（低代码卖点：点选即可改配置）
  - field: region
    kind: multiselect
```

- 未知 op / 缺字段 → 抛出带字段名的清晰错误。

### app/lowcode/pipeline.py

- 支持算子：`filter`（简单表达式 col op value）、`derive`（date 派生 month/weekday）、`aggregate`（group_by + sum/mean/count/min/max）。
- `run_pipeline(df, transforms) -> dict[str, DataFrame]`，`raw` 始终为原始数据。

### app/lowcode/dashboard.py

- `render_dashboard(cfg: PipelineConfig, datasets: dict)`：按 charts 配置画 Plotly，套上 filters 控件；无任何硬编码业务逻辑——这就是「低代码」：改 YAML 即改页面。

### app/kb/*

- chunker：按标题/空行切段，目标 200-500 字，重叠 50 字；每块带 `source`、`heading`。
- vectorstore：numpy 余弦相似度，`build(docs)` / `query(text, k=4)`，持久化到 `.cache/vectors.npz` + json 元数据。
- rag：检索 top-k → 拼 prompt → chat → 回答附「参考来源」列表；prompt 模板放模块常量，评测可调。

### app/analysis/*

- profile：`profile_csv(path) -> dict`：形状、dtypes、缺失率、数值列 describe、重复行率、Top 类别分布。
- nl2sql：LLM 生成 SQL（只允许 SELECT，校验：必须以 SELECT 开头、无分号后追加语句、表名限定为已注册表）→ DuckDB 对 CSV 执行 → 返回 df；失败时把报错回喂 LLM 重试 1 次。

### eval/harness.py（AI Harness 卖点）

- 用例 JSONL 格式：
  - nl2sql.jsonl: `{"id","question","expect_columns":[...],"expect_sql_contains":[...],"gold_sql"}`
  - rag.jsonl: `{"id","question","expect_source_contains":"文件名","expect_answer_keywords":[...] }`
- 指标：
  - NL2SQL：SQL 执行成功率、结果列匹配率、与 gold_sql 执行结果行集一致率
  - RAG：top-k 检索命中期望来源的命中率、回答关键词召回率
- `python -m eval.harness run [--suite all|nl2sql|rag]` → 写 `eval/reports/report-<ts>.json` 与控制台 markdown 摘要（每条用例 PASS/FAIL + 总分明细）。
- harness 必须能在 mock 模式跑通（CI 友好），这是核心卖点：给 AI 功能建立可回归的评测底座。

### app/main.py（Streamlit）

三页（st.navigation 或 selectbox）：
1. 📊 数据分析：上传/示例 CSV → 展示 profile；输入自然语言 → NL2SQL 结果表 + 图
2. 📚 知识库：选文档入库 → 问答（显示引用来源）
3. 🧩 低代码看板：左侧 YAML 编辑器（编辑 configs/*.yaml）→ 右侧即时渲染看板；内置"新建配置"模板按钮

顶栏显示当前 LLM 模式（mock / dashscope）。

## 非功能要求

- `pip install -r requirements.txt` + `streamlit run app/main.py` 两步可跑；无 API Key 全自动 mock。
- 所有单测用 pytest，`pytest -q` 全绿；不依赖网络。
- 代码注释中文；README 英文项目名 + 中文说明。
- requirements.txt 上限：streamlit, pandas, duckdb, plotly, pyyaml, pydantic, dashscope, pytest, python-dotenv。不引入向量数据库/ORM/前端框架。
- 禁止：调用外部网络（单测与 mock 模式下）、写死任何 API Key。
