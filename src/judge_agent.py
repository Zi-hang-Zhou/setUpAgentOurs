"""
法官 Agent（单次 LLM 调用，无容器权限）
职责：根据 Setup 轨迹 + Verifier 对话 + 检察官起诉书，作出最终裁决
"""

import json
import re

from .llm_engine import ARKClient, OpenAICompatibleClient
from .config import get_config
from .logger import get_logger
from .models import ProsecutionResult

logger = get_logger("judge")

SYSTEM_PROMPT = """\
你是法官，负责对 Setup Agent 的工作质量作出最终裁决。

你收到三份材料：
1. Setup Agent 执行轨迹（最近20步）
2. in-loop Verifier 验证对话（Verifier 是 Setup Agent 自己调用的子代理，存在立场偏向）
3. 独立检察官的调查报告（检察官有容器访问权，独立运行测试取证）

## 裁决要点

**关于 Verifier 的可信度**：
Verifier 是 Setup Agent 的内部子代理，天然倾向于放行。若检察官的实际测试结果与 Verifier 的描述出现矛盾，
优先采信检察官的直接证据（命令 + 原始输出）。

**关于 ImportError / ModuleNotFoundError**：
这是最关键的失败类型。核查检察官是否确认该包出现在项目依赖声明（pyproject.toml / setup.cfg / requirements.txt）中：
- 在依赖声明中 → Setup 失职 → guilty
- 仅在可选 extras 中 → 视情形
- 不在依赖声明中 → 项目固有限制 → not_guilty

**关于外部服务 / 测试 bug**：
数据库、API、网络不可用，或纯断言逻辑错误，不是 Setup 的责任，倾向 not_guilty。

**检察官选择不起诉时**：
检察官若未起诉，说明实际运行测试后未发现可追责问题，此时裁定 not_guilty。
但若检察官调查明显不充分（未实际运行测试就结案），可在 reasoning 中注明保留意见。

## 输出格式（严格 JSON）

{"verdict": "not_guilty", "reasoning": "简明裁决依据，100字以内"}
{"verdict": "guilty", "reasoning": "简明裁决依据，点出具体问题，100字以内"}
"""


class JudgeAgent:
    """法官，单次 LLM 调用，无容器权限"""

    def __init__(
        self,
        setup_history: list[dict],
        verify_messages: list[dict],
        prosecution: ProsecutionResult,
    ):
        self._setup_history = setup_history
        self._verify_messages = verify_messages
        self._prosecution = prosecution
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

    def rule(self) -> dict:
        """作出裁决，返回 {"verdict": "guilty"|"not_guilty", "reasoning": "..."}"""
        logger.info("法官开始审阅材料")

        setup_summary = self._format_setup_history()
        verify_summary = self._format_verify_messages()
        prosecution_summary = self._format_prosecution()

        user_content = (
            f"## 一、Setup Agent 执行轨迹（最近20步）\n\n{setup_summary}\n\n"
            f"## 二、in-loop Verifier 验证对话\n\n{verify_summary}\n\n"
            f"## 三、检察官调查报告\n\n{prosecution_summary}\n\n"
            "请根据以上材料作出裁决。"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._llm.chat(messages, json_mode=True)
        logger.info(f"法官裁决原始输出: {raw[:500]}")
        self._llm.close()

        try:
            result = self._parse_json(raw)
            verdict = result.get("verdict", "not_guilty")
            reasoning = result.get("reasoning", "")
            logger.info(f"裁决: verdict={verdict}, reasoning={reasoning[:100]}")
            return {"verdict": verdict, "reasoning": reasoning}
        except Exception as e:
            logger.error(f"法官裁决解析失败: {e}，默认 not_guilty")
            return {"verdict": "not_guilty", "reasoning": f"裁决解析失败: {e}"}

    def _format_setup_history(self) -> str:
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
        if not self._verify_messages:
            return "（无 Verifier 对话记录）"
        lines = []
        for msg in self._verify_messages:
            role = msg.get("role", "?")
            content = (msg.get("content") or "")[:300]
            lines.append(f"[{role}] {content}")
        return "\n\n".join(lines)

    def _format_prosecution(self) -> str:
        if not self._prosecution.prosecute:
            return "检察官选择不起诉：未发现实质性问题。"
        lines = ["检察官提起诉讼，指控如下：\n"]
        for i, charge in enumerate(self._prosecution.charges, 1):
            claim = charge.get("claim", "")
            evidence = charge.get("evidence", "")
            lines.append(f"**指控{i}**：{claim}\n证据：\n{evidence}\n")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(raw: str) -> dict:
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
