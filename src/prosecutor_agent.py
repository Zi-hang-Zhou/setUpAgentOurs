"""
检察官 Agent（ReAct 风格）
职责：调查 Setup Agent 配置的环境是否存在实质性问题
- 有容器访问权，可执行命令取证
- 如果发现问题，提出带具体证据的指控
- 如果没有问题，选择不起诉
- 不安装包、不修改环境
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .environment_manager import EnvironmentManager
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("prosecutor")

MAX_STEPS = 30

SYSTEM_PROMPT = """\
你是检察官，你的任务是对 Verifier 的 success=True 结论进行对抗性核查。
你的立场是怀疑论者：Verifier 有可能说谎或判断失误，你要用实际证据验证。

你有 Setup Agent 执行历史和 Verifier 验证对话作为背景材料。
你在同一个容器内，可以直接执行命令取证。

## 强制调查流程（按顺序执行，不可跳过）

**第一步（必须）：亲自运行测试，获取原始结果**
```
cd /workspace/repo && python3 -m pytest --tb=short -q 2>&1 | tail -30
```
或按 Verifier 提示的方式运行。记录：通过数、失败数、具体错误类型。

**第二步：审查每一种失败原因**
对测试中出现的每类错误，逐一判断：
- `ImportError` / `ModuleNotFoundError`：检查该包是否在项目依赖声明中（pyproject.toml / setup.cfg / requirements.txt）。**若在依赖声明中却未安装，这是 Setup 失职，必须提起诉讼。**
- 版本冲突（`AttributeError` / `TypeError` 在 import 后立即出现）：检查已安装版本与项目要求是否匹配。
- 外部服务不可用（数据库、API、网络）：可以免责。
- 纯测试逻辑断言失败（`AssertionError` 在业务逻辑内）：可以免责。

**第三步：与 Verifier 的报告交叉核验**
Verifier 声称 success=True，你的实际运行结果是否与之一致？
- 若你的测试结果和 Verifier 报告的失败原因相符，且失败都属于免责情形 → 可不起诉
- 若发现 Verifier 遗漏了可追责的失败 → 必须起诉

## 判断标准（严格版）

**必须提起诉讼**：
- 有 `ImportError` / `ModuleNotFoundError`，且该包出现在项目依赖声明中
- 包安装了但版本与项目要求不兼容，导致 import 后即崩溃
- PYTHONPATH / 包路径配置错误，导致项目自身无法导入

**可以不起诉**：
- 失败仅来自外部服务（数据库、Redis、网络请求等）不可用
- 失败是测试本身的逻辑 bug（断言写错、平台差异等），与依赖无关
- 可选 extra 依赖（如 `extras_require` 中的非默认组）未安装

每条指控必须包含：运行的具体命令 + 命令原始输出 + 在依赖声明中的证据。

## 工具（每步只能调用一个，必须响应合法 JSON）

{"thought": "当前观察和下一步推理", "action": "exec_run", "args": {"command": "shell 命令"}}
{"thought": "调查完毕，所有失败均属免责情形", "action": "finish", "args": {"prosecute": false}}
{"thought": "发现可追责问题，提出指控", "action": "finish", "args": {
  "prosecute": true,
  "charges": [
    {"claim": "指控说明（含依赖声明出处）", "evidence": "运行命令: xxx\\n命令输出: yyy"}
  ]
}}

## 硬性约束

- **不安装任何包**：禁止 pip install、apt install 等
- **不修改任何环境配置和项目文件**
"""


class ProsecutorAgent:
    """检察官 ReAct sub-agent，有容器访问权"""

    def __init__(
        self,
        env: EnvironmentManager,
        setup_history: list[dict],
        verify_messages: list[dict],
    ):
        self._env = env
        self._setup_history = setup_history
        self._verify_messages = verify_messages
        self._llm = self._build_llm_client()

    def _build_llm_client(self):
        config = get_config()
        if config.llm_provider == "ark":
            return ARKClient(config.ark)
        elif config.llm_provider == "openai":
            if config.openai is None:
                raise ValueError("LLM_PROVIDER=openai 但未配置 OPENAI_API_KEY")
            return OpenAICompatibleClient(config.openai)
        else:
            raise ValueError(f"不支持的 LLM 提供商: {config.llm_provider}")

    def investigate(self) -> ProsecutionResult:
        """执行调查，返回 ProsecutionResult"""
        logger.info("检察官开始调查")

        # 构造调查背景
        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()

        first_user_msg = (
            f"## Setup Agent 执行轨迹（最近20步）\n\n{setup_summary}\n\n"
            f"## in-loop Verifier 验证对话\n\n{verify_summary}\n\n"
            "请开始调查，判断 Setup Agent 配置的环境是否存在实质性问题。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": first_user_msg},
        ]

        for step in range(1, MAX_STEPS + 1):
            logger.info(f"=== Prosecutor Step {step}/{MAX_STEPS} ===")

            raw = self._llm.chat(messages, json_mode=True)
            logger.info(f"LLM 输出: {raw[:300]}")
            messages.append({"role": "assistant", "content": raw})

            try:
                parsed = self._parse_json(raw)
            except Exception as e:
                obs = f"JSON 解析失败: {e}，请重新输出合法 JSON。"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})
                continue

            action = parsed.get("action", "")
            args = parsed.get("args", {})
            thought = parsed.get("thought", "")
            logger.info(f"action={action}, thought={thought[:80]}")

            if action == "finish":
                prosecute = bool(args.get("prosecute", False))
                charges = args.get("charges", [])
                logger.info(f"调查完成: prosecute={prosecute}, 指控数={len(charges)}")
                self._llm.close()
                return ProsecutionResult(
                    prosecute=prosecute,
                    charges=charges,
                    messages=list(messages),
                )

            elif action == "exec_run":
                cmd = args.get("command", "")
                if not cmd:
                    obs = "错误：exec_run 缺少 command 参数"
                else:
                    result = self._env.exec_run(cmd)
                    obs = (
                        f"exit_code={result.exit_code}\n"
                        f"stdout:\n{result.stdout}\n"
                        f"stderr:\n{result.stderr}"
                    )
                    logger.debug(f"exec_run [{cmd}] → exit_code={result.exit_code}")
                messages.append({"role": "user", "content": f"命令结果:\n{obs}"})

            else:
                obs = f"未知 action='{action}'，只能使用 exec_run / finish"
                logger.warning(obs)
                messages.append({"role": "user", "content": obs})

        logger.warning("Prosecutor 达到最大步数，默认不起诉")
        self._llm.close()
        return ProsecutionResult(
            prosecute=False,
            charges=[],
            messages=list(messages),
        )

    def _format_setup_history(self) -> str:
        """格式化 Setup 历史（最近20步）"""
        recent = self._setup_history[-20:]
        lines = []
        for entry in recent:
            step = entry.get("step", "?")
            action = entry.get("action", {})
            result = entry.get("result", {})
            action_type = action.get("action_type", "?")
            content = action.get("content", {})
            thought = action.get("thought", "")[:100]
            exit_code = result.get("exit_code", "?")
            stdout = (result.get("stdout") or "")[:200]
            lines.append(
                f"[步骤{step}] {action_type} | thought: {thought}\n"
                f"  内容: {json.dumps(content, ensure_ascii=False)[:150]}\n"
                f"  结果: exit_code={exit_code}, stdout: {stdout}"
            )
        return "\n\n".join(lines) if lines else "（无历史）"

    def _format_verify_messages(self) -> str:
        """格式化 Verifier 对话"""
        if not self._verify_messages:
            return "（无 Verifier 对话记录）"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """宽松解析 LLM 输出中的 JSON"""
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"无法提取 JSON: {raw[:200]}")
