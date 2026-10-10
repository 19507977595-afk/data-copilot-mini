# 里程碑计划（给 Claude Code 的施工单）

每完成一个里程碑：运行 `pytest -q` 确认全绿，并把改动 `git add -A && git commit -m "M<n>: <摘要>"`。
不要跳序施工；SPEC.md 是唯一事实来源，与本文件冲突时以 SPEC.md 为准。

## M1 骨架 + LLM 客户端 + 低代码配置核心
- 建目录结构、requirements.txt、.env.example、.gitignore（忽略 .cache/ logs/ eval/reports/ __pycache__/）
- 生成 data/sales.csv（500+ 行，用固定随机种子 python 脚本生成后删脚本或留 scripts/gen_data.py）与 data/kb/ 3 篇中文 Markdown
- 实现 app/llm/client.py（含 mock、traces 日志）
- 实现 app/config.py + configs/sales_dashboard.yaml
- tests: test_config.py；pytest 全绿
验收：`pytest -q` 通过；`python -c "from app.llm.client import LLMClient; c=LLMClient(); assert c.mock; print(c.embed(['a','b'])[0][:4])"` 可运行

## M2 数据分析（profile + NL2SQL）
- app/analysis/profile.py、app/analysis/nl2sql.py（含 SQL 安全校验与一次重试）
- eval/cases/nl2sql.jsonl >= 6 条用例
- tests: test_profile.py
验收：pytest 全绿；`python -m eval.harness run --suite nl2sql` 在 mock 模式可出报告（harness 本体可先在 M2 落地最小版或留到 M5，但用例文件必须就位）

## M3 知识库 RAG
- app/kb/chunker.py、vectorstore.py、rag.py
- eval/cases/rag.jsonl >= 6 条用例
- tests: test_chunker.py、test_vectorstore.py
验收：pytest 全绿；mock 模式下 `rag.answer("退货期限是多久？")` 返回含来源的回答

## M4 Streamlit UI
- app/main.py 三页 + lowcode/pipeline.py + lowcode/dashboard.py 页面整合（pipeline/dashboard 若前序未建则本里程碑建）
- 无头可导入性：`python -c "import app.main"` 不报错（Streamlit 运行时代码放 if 块或函数内）
验收：pytest 全绿 + 上述 import 通过

## M5 AI Harness 完整版 + README
- eval/harness.py 完整：两套件、指标聚合、JSON+控制台报告
- tests: test_harness.py
- README.md：项目定位、四大卖点（低代码/知识库/数据分析/AI Harness）、架构图（mermaid）、快速开始、mock 与真实模式说明、评测命令示例
验收：`pytest -q` 全绿；`python -m eval.harness run`（默认 all）生成 eval/reports/*.json
