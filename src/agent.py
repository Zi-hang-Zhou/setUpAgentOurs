"""
主控制器（按 blueprint 1.4 和 2 节定义）
实现推测执行的 Setup Agent
"""

import time
from typing import Any

from .config import get_config
from .logger import get_logger
from .models import (
    AgentState,
    AgentAction,
    ActionType,
    CommandResult,
    AttributionReport,
    XPUSuggestion,
)
from .environment_manager import EnvironmentManager
from .xpu_client import create_xpu_client, XPUClientBase
from .llm_engine import LLMEngine

logger = get_logger("agent")


class SpeculativeSetupAgent:
    """推测执行的环境配置 Agent（按 blueprint 1.4 节定义）"""

    def __init__(self, repo_url: str, max_steps: int = 50):
        """初始化 Agent

        Args:
            repo_url: 目标 Git 仓库 URL
            max_steps: 最大迭代次数
        """
        self._state = AgentState(
            repo_url=repo_url,
            max_steps=max_steps,
        )
        self._env: EnvironmentManager = EnvironmentManager()
        self._xpu: XPUClientBase = create_xpu_client()
        self._llm: LLMEngine = LLMEngine()

        # 缓存 XPU 建议
        self._current_xpu_suggestions: list[XPUSuggestion] = []

        logger.info(f"Agent 初始化完成，目标仓库: {repo_url}")

    def run(self) -> str:
        """运行 Agent 主循环（按 blueprint 2 节实现）"""
        logger.info("开始执行环境配置任务")

        # 1. 初始化：创建容器并克隆仓库
        container_id = self._env.create_container(self._state.repo_url)
        self._state.container_id = container_id

        while self._state.step < self._state.max_steps:
            self._state.step += 1
            logger.info(f"=== Step {self._state.step}/{self._state.max_steps} ===")

            # 2. 观测 (Observation)
            cwd = self._env.exec_run("pwd").stdout.strip()
            os_info = self._env.exec_run("cat /etc/os-release | head -2").stdout.strip()

            # 3. 诊断与检索 (Diagnosis & Retrieval)
            self._current_xpu_suggestions = []
            if self._state.last_error:
                context = {
                    "error": self._state.last_error,
                    "os_release": os_info,
                }
                self._current_xpu_suggestions = self._xpu.query(context)

            # 4. 思考 (Thought & Plan) - LLM 决策
            action = self._llm.generate_action(
                history=self._state.get_recent_history(),
                xpu_suggestions=self._current_xpu_suggestions,
                cwd=cwd,
                os_info=os_info,
                last_error=self._state.last_error,
                failed_suggestion_ids=self._state.failed_suggestions,
            )

            logger.info(f"决策: {action}")

            # 5. 执行 (Execution)
            if action.action_type == ActionType.SHELL_COMMAND:
                self._handle_shell_command(action)

            elif action.action_type == ActionType.TRY_XPU_SUGGESTION:
                self._handle_try_xpu_suggestion(action)

            elif action.action_type == ActionType.SET_ENV:
                self._handle_set_env(action)

            elif action.action_type == ActionType.FINISH:
                self._handle_finish(action)
                break

        # 清理资源
        self._cleanup()

        if self._state.completed:
            logger.info(f"任务完成: {self._state.final_message}")
            return self._state.final_message or "任务完成"
        else:
            logger.warning("达到最大迭代次数，任务未完成")
            return f"达到最大迭代次数 ({self._state.max_steps})，任务未完成"

    def _handle_shell_command(self, action: AgentAction) -> None:
        """处理 SHELL_COMMAND 动作"""
        if not action.command:
            logger.warning("SHELL_COMMAND 缺少 command")
            return

        result = self._env.exec_run(action.command)

        # 更新错误状态
        if not result.success:
            self._state.last_error = result.stderr or result.stdout
        else:
            self._state.last_error = None

        # 记录历史
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": result.to_dict(),
        })

    def _handle_try_xpu_suggestion(self, action: AgentAction) -> None:
        """处理 TRY_XPU_SUGGESTION 动作（推测执行模式，按 blueprint 2 节实现）"""
        if not action.xpu_suggestion_id:
            logger.warning("TRY_XPU_SUGGESTION 缺少 xpu_suggestion_id")
            return

        # 查找对应的 XPU 建议
        suggestion = None
        for s in self._current_xpu_suggestions:
            if s.id == action.xpu_suggestion_id:
                suggestion = s
                break

        if not suggestion:
            logger.warning(f"未找到 XPU 建议: {action.xpu_suggestion_id}")
            return

        # 记录执行前的错误
        error_before = self._state.last_error or ""

        # A. 存档 (Checkpoint)
        ckpt_tag = f"step_{self._state.step}_pre_xpu"
        self._env.create_checkpoint(ckpt_tag)
        logger.info(f"创建快照 {ckpt_tag}，开始推测执行 XPU 建议")

        # B. 试错 (Trial)
        success = True
        logs: list[CommandResult] = []
        for cmd in suggestion.commands:
            result = self._env.exec_run(cmd)
            logs.append(result)
            if not result.success:
                success = False
                break

        # C. 验证与归因 (Verification & Attribution)
        error_after = ""
        if not success and logs:
            error_after = logs[-1].stderr or logs[-1].stdout

        # 计算归因分数
        if success:
            attribution_score = 1.0
            outcome = "SUCCESS"
        elif error_after and error_after != error_before:
            attribution_score = -1.0  # 导致新错误
            outcome = "FAIL"
        else:
            attribution_score = 0.0  # 无效果
            outcome = "FAIL"

        # D. 提交反馈 (Feedback Loop)
        report = AttributionReport(
            suggestion_id=suggestion.id,
            timestamp=time.time(),
            repo_context=self._state.repo_url,
            outcome=outcome,
            error_before=error_before,
            error_after=error_after,
            score=attribution_score,
            logs=logs,
        )
        self._xpu.submit_feedback(report)

        # E. 决策分支 (Decision Branch)
        if not success:
            logger.info(f"XPU 建议 {suggestion.id} 执行失败，回滚中...")
            self._env.rollback_to_checkpoint()
            # 记录失败，防止 LLM 重试
            self._state.record_failed_suggestion(suggestion.id)
            self._state.last_error = error_before  # 恢复原错误
        else:
            logger.info(f"XPU 建议 {suggestion.id} 验证通过")
            self._state.last_error = None

        # 记录历史
        self._state.add_to_history({
            "action": action.to_dict(),
            "xpu_suggestion": suggestion.to_dict(),
            "outcome": outcome,
            "attribution_score": attribution_score,
            "logs": [log.to_dict() for log in logs],
        })

    def _handle_set_env(self, action: AgentAction) -> None:
        """处理 SET_ENV 动作"""
        if not action.env_key or action.env_value is None:
            logger.warning("SET_ENV 缺少 env_key 或 env_value")
            return

        self._env.set_env(action.env_key, action.env_value)
        self._state.env_vars[action.env_key] = action.env_value

        self._state.add_to_history({
            "action": action.to_dict(),
        })

    def _handle_finish(self, action: AgentAction) -> None:
        """处理 FINISH 动作"""
        self._state.completed = True
        self._state.final_message = action.message or "任务完成"

        self._state.add_to_history({
            "action": action.to_dict(),
        })

    def _cleanup(self) -> None:
        """清理资源"""
        logger.info("清理 Agent 资源...")
        self._env.cleanup()
        self._llm.close()

        if hasattr(self._xpu, "close"):
            self._xpu.close()

        logger.info("资源清理完成")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup()
        return False
