# 项目交接文档（服务器 Claude 请完整阅读）

## 0. 编码规范（最高优先级，必须遵守）

- **语言**：注释、提示信息、日志、报错一律中文；代码字符串里不出现英文提示；永远用中文与我交流
- **配置与日志**：整个项目只保留一套配置注入（`src/config.py`）和一套日志系统（`src/logger.py`），禁止每个脚本自己写 `echo` 或 `log()`
- **错误处理**：出错立即退出，返回非零，保留原始错误信息；禁用 `|| true`、`2>/dev/null`、`set +e` 等掩盖错误的做法；不捕获、不兜底、不静默重试
- **输出留痕**：默认把 stdout 和 stderr 重定向到 `log/$(date +%Y%m%d-%H%M%S).log`；屏幕仅输出必要日志（ERROR 级别）
- **开发环境**：永远优先使用 `.venv/` 内的虚拟解释器，严禁在全局解释器下 pip 安装任何包
- **脚本粒度**：能用 bash 单行解决就不写脚本；必须写时，一个脚本只干一件事；不做多余封装
- **简洁**：不加没被要求的功能；不写多余注释；不做防御性兼容

---

## 1. 项目目标

**Speculative Setup Agent** —— 自动为任意 Python 仓库搭建可运行的测试环境。

给定一个 GitHub 仓库 URL，Agent 在隔离 Docker 容器内自主执行：克隆代码 → 分析依赖 → 安装环境 → 运行 pytest 验证。整个过程无人工干预，最终输出"环境是否配置成功"的裁决。

目标场景：大规模批量评测（如 EnvBench benchmark 的 329 个 Python 仓库）。

---

## 2. 核心设计理念

### 三阶段流水线（`src/main.py` 编排）

```
阶段1 Setup     → agent.run()           → SetupResult（容器保留）
阶段2 Phase 2   → Prosecutor + Judge    → 诉讼裁决
XPU提取         → _store_xpu_experience → 入向量库
阶段3 Report    → 写 JSON 结果文件（log/<repo>_result.json）
```

**Phase 2 诉讼模型**是本项目的核心创新：Setup Agent 宣布 FINISH 后，独立的 Prosecutor（检察官）进入容器复查，发现问题则提起诉讼；Judge（法官）综合证据做最终裁决（guilty/not_guilty）。把"Agent 自我报告"和"独立核查"分离，避免自说自话。

### XPU（eXPerience Unit）经验知识库

每次任务完成后，将本次轨迹 + Phase 2 诉讼证据（charges）+ 裁决推理一起喂给 LLM，提炼可复用的**工具/框架层面泛化经验**（而非仓库特定事实），存入 PostgreSQL 向量数据库。下次遇到相似问题时，Agent 优先检索 XPU 建议，实现在线学习。

关键约束：XPU 只记录泛化规律（如"pyproject.toml 含 [tool.poetry] 时必须用 poetry install"），禁止记录"该仓库需要包X"这类特定事实。

### 推测执行

Agent 看到 XPU 建议后，用 `TRY_XPU_SUGGESTION` 动作推测执行，失败则回滚容器到快照，避免错误操作污染环境。

---

## 3. 代码结构

```
src/
├── main.py                 # 三阶段编排入口，含 _store_xpu_experience() / _build_traj_from_history()
├── agent.py                # Setup Agent，LLM 驱动的环境配置主循环
├── prosecutor_agent.py     # Prosecutor：独立复查，提起诉讼
├── judge_agent.py          # Judge：综合证据最终裁决
├── verifier_agent.py       # Verifier sub-agent：pytest 两步验证（collect + run）
├── environment_manager.py  # Docker 容器生命周期（cleanup_snapshots/destroy/cleanup）
├── llm_engine.py           # LLM 客户端（ARK / OpenAI 兼容双模式）
├── models.py               # 数据类：SetupResult, ProsecutionResult, Phase2Result 等
├── config.py               # 配置加载（统一读 .env）
├── logger.py               # 统一日志系统
├── xpu_client.py           # XPU 查询入口（向量检索 + mock 模式）
└── xpu/
    ├── xpu_vector_store.py              # PostgreSQL + pgvector 存取，ThreadedConnectionPool
    ├── xpu_adapter.py                   # XpuEntry / XpuAtom 数据结构
    ├── xpu_dedup.py                     # 去重逻辑（LLM 判断是否合并已有条目）
    └── extract_xpu_from_trajs_mvp.py   # LLM 提取 XPU 经验，支持 phase2_context 参数

scripts/
├── run_repo_list.py        # 批量运行主脚本（多 worker + 进度条）
├── import_xpu_jsonl.py     # 从 JSONL 文件批量导入 XPU 到数据库
└── reset_db.py             # 重置数据库表结构（危险：会清空数据）

data/
├── python329.jsonl         # 完整 329 仓库评测集（EnvBench benchmark）
├── python8_329.jsonl       # 8 个测试仓库子集
└── two_crashed.jsonl       # 2 个之前因磁盘满崩溃需重跑的仓库

blueprint.md        # 原始架构设计文档（早期草稿，部分已过时，仅供历史参考）
INTEGRATION_DOC.md  # XPU 集成说明（早期，仅供历史参考）
```

---

## 4. 外部依赖与服务配置

### Docker（必须）
容器内运行 `ubuntu:22.04`，每个仓库独立容器，任务结束后销毁。
```bash
docker info  # 确认 Docker daemon 正常运行
```

### PostgreSQL + pgvector（必须）

数据库名 `xpu26`，存储 XPU 经验向量（1024 维，DashScope text-embedding-v4）。

建库步骤（若从零开始）：
```sql
CREATE DATABASE xpu26;
\c xpu26
CREATE EXTENSION IF NOT EXISTS vector;
```
然后执行 `python scripts/reset_db.py` 建表。

**迁移现有数据**（推荐）：
```bash
# 将本地导出的 xpu26_export.sql 传到服务器后：
psql postgresql://postgres:password@localhost:5432/xpu26 < xpu26_export.sql
```
`xpu26_export.sql` 包含 163 条 XPU 经验，约 2MB，不进 git，需单独传输。

### Python 环境
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### .env 配置（不进 git，需手动创建）

```env
# LLM 主流程（DeepSeek）
LLM_PROVIDER=openai
OPENAI_API_KEY=<DeepSeek API Key>
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat

# Embedding（DashScope，XPU 向量化用）
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=<DashScope API Key>
EMBEDDING_DIM=1024

# XPU 经验提取 LLM（Phase 2 结束后触发，同样用 DeepSeek）
XPU_EXTRACT_API_KEY_ENV=OPENAI_API_KEY
XPU_EXTRACT_BASE_URL_ENV=OPENAI_BASE_URL
XPU_EXTRACT_MODEL=deepseek-chat
XPU_EXTRACT_TIMEOUT=300

# Docker
DOCKER_BASE_IMAGE=ubuntu:22.04
DOCKER_WORK_DIR=/workspace
DOCKER_TIMEOUT=300

# PostgreSQL 连接串（键名必须是 dns，历史遗留，不要改）
dns=postgresql://postgres:password@localhost:5432/xpu26
XPU_VECTOR_ENABLED=true
XPU_ENABLED=true
```

---

## 5. 运行方式

### 单个仓库
```bash
.venv/bin/python -m src.main https://github.com/user/repo [max_steps]
```

### 批量运行
```bash
.venv/bin/python scripts/run_repo_list.py \
  --list data/python8_329.jsonl \
  --workers 3 \
  --max-steps 50 \
  --output-dir runs 2>&1 | tee runs/batch_run.log
```

`--workers` 建议值：服务器 RAM ≥ 32GB 可用 4，16GB 用 3，8GB 用 2。每个 agent 容器峰值约 1.5~2GB。

---

## 6. 踩过的坑与经验

### XPU 提取时序问题（已修复）
提取原来在 `agent.py` 的 `run()` 末尾触发，Phase 2 未运行，诉讼证据全丢。已修复：移到 `main.py`，Phase 2 全部完成后才触发 `_store_xpu_experience()`，同时把 `prosecution.charges` 和 `judgment.reasoning` 一并传入。

### 容器磁盘满导致批量崩溃
停止的容器 + 悬空镜像积累，写日志时 `OSError: No space left on device`。
定期执行：`docker system prune -f`

### XPU 提炼质量
早期容易记录"该仓库需要包X"这类无用特定事实。已在 `extract_xpu_from_trajs_mvp.py` system_text 中加入【提炼原则】段落，强制泛化。

### `dns` 环境变量名
PostgreSQL 连接串的键名是 `dns`（历史遗留），不是 `DATABASE_URL`，不要改。

### ROLLBACK_ENV 快照回滚
若容器没有可用快照时回滚会抛异常，已修复为返回错误信息而非崩溃。

### Docker Desktop 内存限制（Mac 特有，服务器无此问题）
Mac 上 Docker Desktop 有独立 VM，默认 7.65GB，被常驻服务（qdrant/neo4j 等）占去约 3.5GB，只剩 4GB 给 agent，并发受限。服务器上 Docker 直接用宿主机内存，无此问题，可以放心开 4 worker。

---

## 7. 当前进展与下一步

- 已对 8 个仓库做测试，XPU 数据库积累 163 条经验
- Phase 2 诉讼模型（Prosecutor + Judge）已集成并验证
- XPU 提取已升级为接收 phase2_context（charges + verdict + reasoning）
- **下一步**：服务器上用 `--workers 3~4` 跑完整 329 仓库集（`python329.jsonl`），观察 guilty 案例的 charges 能否提炼出高质量 XPU
