# 工程落地指南：Speculative Setup Agent (Native XPU Support)

**文档目标**：构建一个基于 Docker 的自动化环境配置智能体（Agent），该智能体能够原生集成外部经验单元（XPU），并通过“推测执行（Speculative Execution）”机制验证检索到的知识。

**核心架构风格**：
*   **底座**：线性执行循环 + Docker 状态快照/回滚（参考 *Repo2Run*）。
*   **推理**：ReAct 循环的变体，增加了“诊断-检索-验证”子流程（参考 *ExecutionAgent* 的 Feedback Loop）。
*   **创新点**：将 XPU 建议视为“假设（Hypothesis）”，在执行前进行环境存档（Checkpoint），执行后进行归因评价（Attribution）。

---

## 1. 系统模块定义 (Class Definitions)

你需要实现以下四个核心类。

### 1.1 `EnvironmentManager` (Docker 交互层)
负责所有与 Docker 的底层交互，特别是**状态管理**。

*   **Attributes**:
    *   `container_id`: str
    *   `image_name`: str
    *   `history_snapshots`: List[str] (存储 commit 后的 image tags，用作栈)

*   **Methods**:
    *   `exec_run(command: str, timeout: int = 300) -> CommandResult`:
        *   执行 Shell 命令。
        *   **关键逻辑**：必须捕获 `exit_code`, `stdout`, `stderr`。
        *   **返回结构**：`dataclass CommandResult(exit_code: int, stdout: str, stderr: str)`
    *   `create_checkpoint(tag: str) -> str`:
        *   执行 `docker commit`，保存当前容器状态为一个新的 Image。
        *   将 tag 压入 `history_snapshots` 栈。
        *   返回 image id。
    *   `rollback_to_checkpoint() -> bool`:
        *   从 `history_snapshots` 弹出最近的 tag。
        *   销毁当前容器。
        *   使用弹出的 image tag 重新启动一个新的容器（保持挂载点/环境变量一致）。
    *   `read_file(path: str) -> str`: 读取容器内文件内容。
    *   `write_file(path: str, content: str) -> bool`: 向容器内写入文件。

### 1.2 `XPUClient` (外部知识接口)
负责与外部 XPU 系统通信。

*   **Methods**:
    *   `query(context: Dict) -> List[XPUSuggestion]`:
        *   **Input**: `context` 包含 `{error_log: str, repo_metadata: str, current_packages: List[str]}`。
        *   **Output**: 返回 Top-K 建议列表。
        *   **Data Structure**:
            ```python
            @dataclass
            class XPUSuggestion:
                id: str
                description: str  # 简短描述，例如 "Downgrade numpy to 1.21"
                commands: List[str] # 具体的 Shell 指令
                confidence: float
            ```
    *   `submit_feedback(report: AttributionReport) -> None`:
        *   用于回传归因报告（定义见下文）。

### 1.3 `LLMEngine` (推理核心)
负责生成指令和解析输出。

*   **Methods**:
    *   `generate_action(history: List[Message], xpu_context: List[XPUSuggestion]) -> AgentAction`:
        *   构造 Prompt，将 `xpu_context` 注入到 System Prompt 的特定 Section 中。
        *   强制 LLM 输出 JSON 格式。

### 1.4 `SpeculativeSetupAgent` (主控制器)
负责编排整个流程。

*   **Attributes**:
    *   `env`: EnvironmentManager
    *   `xpu`: XPUClient
    *   `llm`: LLMEngine
    *   `max_steps`: int

---

## 2. 核心控制流逻辑 (Main Loop Implementation)

这是 Agent 的主循环逻辑，必须严格按照以下伪代码实现 `run()` 方法。

```python
def run(self, repo_url: str):
    # 1. 初始化
    self.env.exec_run(f"git clone {repo_url}")
    
    step = 0
    while step < self.max_steps:
        # 2. 观测 (Observation)
        # 获取当前目录、最近的报错日志、已安装包列表
        cwd = self.env.exec_run("pwd").stdout
        last_error = self._extract_last_error() 
        
        # 3. 诊断与检索 (Diagnosis & Retrieval) - 原生支持 XPU 的关键
        xpu_suggestions = []
        if last_error:
            # 主动向 XPU 查询：这个错误以前见过吗？
            context = {
                "error": last_error, 
                "os_release": self.env.exec_run("cat /etc/os-release").stdout
            }
            xpu_suggestions = self.xpu.query(context)
        
        # 4. 思考 (Thought & Plan)
        # 将 XPU 建议作为 context 传给 LLM
        # LLM 决定是采用 XPU 建议，还是自己生成命令
        action = self.llm.generate_action(self.history, xpu_suggestions)
        
        # 5. 执行 (Execution)
        if action.type == "SHELL_COMMAND":
            # 常规模式：直接执行，不回滚（除非 fatal error）
            result = self.env.exec_run(action.command)
            self._record_history(action, result)
            
        elif action.type == "TRY_XPU_SUGGESTION":
            # === 推测执行模式 (Speculative Mode) ===
            suggestion = next(s for s in xpu_suggestions if s.id == action.xpu_id)
            
            # A. 存档 (Checkpoint)
            ckpt_id = self.env.create_checkpoint(f"step_{step}_pre_xpu")
            
            # B. 试错 (Trial)
            success = True
            logs = []
            for cmd in suggestion.commands:
                res = self.env.exec_run(cmd)
                logs.append(res)
                if res.exit_code != 0:
                    success = False
                    break
            
            # C. 验证与归因 (Verification & Attribution)
            # 运行一个轻量级验证（如 python -c "import pkg"）或检查 exit code
            attribution_score = 1.0 if success else -1.0
            
            # D. 提交反馈 (Feedback Loop)
            report = AttributionReport(
                suggestion_id=suggestion.id,
                outcome="SUCCESS" if success else "FAIL",
                logs=logs,
                score=attribution_score
            )
            self.xpu.submit_feedback(report)
            
            # E. 决策分支 (Decision Branch)
            if not success:
                print(f"XPU Suggestion {suggestion.id} failed. Rolling back.")
                self.env.rollback_to_checkpoint()
                # 将此失败记录添加到历史，防止 LLM 重试同一个建议
                self._record_failure_context(suggestion.id)
            else:
                print(f"XPU Suggestion {suggestion.id} verified.")
                # 继续，不回滚
        
        step += 1
```

---

## 3. 数据协议与 Prompt 设计

### 3.1 LLM 输出协议 (JSON Schema)
Agent 的 `LLMEngine` 必须强制模型返回以下 JSON 格式。不要使用自由文本。

```json
{
  "thought": "分析当前状态和错误原因...",
  "action_type": "SHELL_COMMAND" | "TRY_XPU_SUGGESTION" | "FINISH",
  "content": {
    // 如果是 SHELL_COMMAND
    "command": "pip install numpy",
    
    // 如果是 TRY_XPU_SUGGESTION
    "xpu_suggestion_id": "suggestion_123",
    "reasoning": "XPU 建议降级 numpy 版本，这与报错信息高度吻合"
  }
}
```

### 3.2 System Prompt 模板
在构建 `LLMEngine` 时，使用以下模板注入 System Message。

```text
You are an expert DevOps agent tailored for environment setup.
You have access to a Linux terminal and an external eXPerience Unit (XPU).

Current Status:
- WorkDir: {cwd}
- OS: {os_info}

XPU Suggestions (Proven solutions from history):
{formatted_xpu_suggestions} 
// 格式如: [ID: 1] Description: Install libmysqlclient-dev. Confidence: 0.9

Instructions:
1. Review the Last Error and XPU Suggestions.
2. If an XPU suggestion seems relevant, PREFER using action_type="TRY_XPU_SUGGESTION" with its ID. This will trigger a safe, verifiable execution sandbox.
3. If no XPU suggestion fits, generate a standard shell command using action_type="SHELL_COMMAND".
4. Always analyze WHY you chose a specific action in the "thought" field.
```

---

## 4. 关键实现细节 Checklist

在编码时，请务必处理以下 Corner Cases：

1.  **输出截断 (Output Truncation)**:
    *   `exec_run` 方法中，如果 stdout 超过 2000 字符，必须进行截断（保留头部 1000 和尾部 1000），并在中间插入 `...[Truncated]...`。否则 LLM 的 Context Window 会爆。
2.  **交互式命令处理**:
    *   Docker 执行 `apt-get` 或 `pip` 时，必须附加 `-y` 或 `--no-input` 参数。如果 Agent 生成了需要交互的命令（如等待用户输入的 `python script.py`），`EnvironmentManager` 必须能检测超时并 kill 进程。
3.  **状态持久化**:
    *   `ENV` 变量在 `docker exec` 之间默认是不持久的。如果 LLM 执行 `export PATH=...`，下一个命令会丢失。
    *   **解决方案**：在 `EnvironmentManager` 中维护一个 `env_vars` 字典，每次 `exec_run` 时显式通过 `docker exec -e KEY=VAL` 注入这些变量。或者，强制 Agent 将环境变量写入 `~/.bashrc` 并每次 source。

---

## 5. 归因报告结构 (Attribution Report)

这是 XPU 系统进行自我进化的关键数据。你的 Agent 必须在每次 `TRY_XPU_SUGGESTION` 后生成此对象。

```python
@dataclass
class AttributionReport:
    suggestion_id: str
    timestamp: float
    repo_context: str  # 当前仓库名/语言
    outcome: str       # "SUCCESS" | "FAIL" | "PARTIAL"
    error_before: str  # 采纳建议前的报错
    error_after: str   # 采纳建议后的报错（如果还有）
    score: float       # 1.0 (解决问题) -> 0.0 (无效果) -> -1.0 (导致新错误)
```