# XPU 经验知识库集成文档

## 一、项目概述

集成了 XPU（eXPerience Unit）经验知识库系统。改造后的 Agent 具备以下能力：

1. **经验检索**：遇到错误时，从向量数据库中搜索历史解决方案
2. **经验驱动决策**：LLM 可以选择直接采用历史经验（TRY_XPU_SUGGESTION），而不是从零推理
3. **在线学习**：任务完成后，自动将本次修复过程提炼为新经验并入库
4. **环境回滚**：新增 ROLLBACK_ENV 动作，容器状态混乱时可回滚到快照

## 二、架构变更总览

```
原始 Agent 流程:
  LLM 决策 → 执行命令 → 观察结果 → 循环

改造后流程:
  XPU 向量搜索 → LLM 决策(含历史建议) → 执行命令 → 观察结果 → 循环
       ↑                                                     |
       |                    任务完成时                              |
       +←←←← LLM 提取经验 ←←← 轨迹转换 ←←← _store_experience ←←←+
```

## 三、文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/xpu/__init__.py` | XPU 模块初始化 |
| `src/xpu/xpu_adapter.py` | XPU 数据结构（XpuEntry, XpuAtom）和评分检索逻辑 |
| `src/xpu/xpu_vector_store.py` | PostgreSQL + pgvector 向量数据库操作（搜索/插入/更新/遥测） |
| `src/xpu/xpu_dedup.py` | LLM 驱动的去重与智能合并（相似度阈值 0.85） |
| `src/xpu/extract_xpu_from_trajs_mvp.py` | 从轨迹中提取 XPU 经验的 LLM 管道 |
| `src/xpu/online_xpu_extractor.py` | 在线提取入口（track.json → JSONL → LLM → 入库） |
| `.env` | 环境配置（LLM / Embedding / 数据库 / Docker） |

### 修改文件

| 文件 | 改动 |
|------|------|
| `src/config.py` | XPUConfig 新增 `db_dns`、`vector_enabled` 字段 |
| `src/xpu_client.py` | 新增 `VectorXPUClient` 类 + 工厂函数优先级调整 |
| `src/models.py` | ActionType 新增 `ROLLBACK_ENV` |
| `src/agent.py` | 新增 `_handle_rollback_env()`、`_store_experience_if_applicable()` |
| `src/llm_engine.py` | 适配推理模型、system prompt 增加新动作说明、解析 ROLLBACK_ENV |
| `src/environment_manager.py` | git clone 改为 HTTP/1.1 + 3 次重试 |

## 四、各模块详细说明

### 4.1 VectorXPUClient（`xpu_client.py`）

替代原有的 MockXPUClient，实现了基于向量相似度的经验检索。

**工厂优先级**：VectorXPUClient > HTTPXPUClient > MockXPUClient

**query() 流程**：
1. 从 context 中提取 error_text
2. 调用 Embedding API 生成向量（text-embedding-3-small, 1536 维）
3. 在 pgvector 中做 KNN 搜索（k=3, min_similarity=0.3）
4. 将 XpuEntry 转换为 XPUSuggestion（id, description, commands）
5. 记录 telemetry hits

**submit_feedback() 流程**：
- score > 0 → increment telemetry successes
- score < 0 → increment telemetry failures

### 4.2 经验在线学习（`agent.py` → `_store_experience_if_applicable()`）

任务 FINISH 后自动触发，复用 `xpu_standalone` 的完整 LLM 提取管道。

**6 步流程**：

| 步骤 | 操作 | 说明 |
|------|------|------|
| 1 | 历史转轨迹 | agent history → Repo2Run JSONL 格式 |
| 2 | 写临时文件 | `{safe_name}@HEAD.jsonl` |
| 3 | LLM 提取 | 调用 `extract_xpu_from_trajs()` |
| 4 | 过滤有效经验 | `llm_decision == 'xpu'` |
| 5 | 构建 XpuEntry | 与 `online_xpu_extractor.py` 完全一致 |
| 6 | 去重入库 | `dedup_and_store(use_llm=True)` |

**轨迹格式转换规则**：
- SHELL_COMMAND → `{"role": "assistant", "content": "执行命令:\n```bash\ncmd\n```"}`
- 命令输出/报错 → `{"role": "system", "content": "stderr or stdout"}`

### 4.3 ROLLBACK_ENV 动作

新增的 LLM 可调用动作，用于环境状态混乱时回滚 Docker 容器到最近快照。

- `models.py`：ActionType 枚举新增 ROLLBACK_ENV
- `llm_engine.py`：system prompt 说明 + 解析逻辑
- `agent.py`：调用 `self._env.rollback_to_checkpoint()`

### 4.4 LLM Engine 适配（`llm_engine.py`）

- **推理模型兼容**：glm-4.6 等推理模型返回 `reasoning_content` 而非 `content`，优先取 content，为空时 fallback 到 reasoning_content
- **max_tokens**：显式设置 4096，避免推理模型 token 不足
- **system prompt**：补充了 SET_ENV、ROLLBACK_ENV 的 JSON schema 说明和使用场景

### 4.5 关于 Git Clone （`environment_manager.py`）

Docker 内 git clone 经常因网络问题失败，改进：
- `git config --global http.version HTTP/1.1`（避免 HTTP2 framing 错误）
- 最多重试 3 次

## 五、配置说明（.env）

```bash
# LLM 决策引擎
LLM_PROVIDER=openai
OPENAI_API_KEY=<key>
OPENAI_BASE_URL=<base_url>
OPENAI_MODEL=glm-4.6

# Embedding 服务（必须支持 /v1/embeddings，维度 1536）
EMBEDDING_API_KEY=<key>
EMBEDDING_BASE_URL=<base_url>
EMBEDDING_MODEL=text-embedding-3-small

# 向量数据库
XPU_VECTOR_ENABLED=true
dns=postgresql://user:pass@host:port/xpu_db

# XPU 经验提取（复用 LLM 配置）
XPU_EXTRACT_API_KEY_ENV=OPENAI_API_KEY
XPU_EXTRACT_BASE_URL_ENV=OPENAI_BASE_URL
XPU_EXTRACT_MODEL=glm-4.6
```

**注意**：LLM 接口和 Embedding 接口可以分开配置，因为很多 LLM 服务不提供 embedding 端点。

## 六、去重逻辑

新经验入库前经过三层判断：

1. **向量相似度 < 0.85** → 直接作为新经验存入
2. **向量相似度 >= 0.85** → LLM 判断是否为同一根因
   - 不同根因 → 作为新经验存入
   - 同一根因 → LLM 智能合并建议后更新原有条目

## 七、端到端验证结果

以 `benthayer/git-gud` 仓库测试，13 步完成部署：

```
Step 1:  ls -la                                    # 探索项目
Step 2:  cat README.md                             # 读文档
Step 3:  python3 --version                         # 检测缺 Python
Step 4:  apt install python3 python3-dev python3-pip  # 安装（XPU 建议）
Step 5:  python3 --version && pip3 --version       # 验证
Step 6:  pip3 install -e .                         # 安装项目
Step 7-12: git gud init / goal / status            # 验证功能
Step 13: FINISH                                    # 完成

→ 自动触发 _store_experience_if_applicable()
→ LLM 提取经验（39 秒）
→ [XPU Store] new: 成功提取并存储（新经验）
→ 新增条目: xpu_env_py_missing（缺少 Python 的修复经验）
```
