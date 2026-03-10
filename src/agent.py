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
    SetupResult,
)
from .environment_manager import EnvironmentManager
from .xpu_client import create_xpu_client, XPUClientBase
from .llm_engine import LLMEngine
from .verifier_agent import VerifierAgent

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
        # 保存最后一次成功 verify 的 Verifier 对话轨迹，供 Phase 2 使用
        self._last_verify_messages: list[dict] = []

        logger.info(f"Agent 初始化完成，目标仓库: {repo_url}")

    @property
    def env(self) -> EnvironmentManager:
        """暴露环境管理器，供验证阶段复用同一容器"""
        return self._env

    def run(self) -> SetupResult:
        """运行 Agent 主循环（按 blueprint 2 节实现）

        返回 SetupResult，容器保留供后续验证阶段使用。
        调用方负责在验证完成后调用 env.destroy() 销毁容器。
        """
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

            elif action.action_type == ActionType.ROLLBACK_ENV:
                self._handle_rollback_env(action)

            elif action.action_type == ActionType.VERIFY:
                verified = self._handle_verify(action)
                if verified:
                    break  # 验证通过，退出主循环

            elif action.action_type == ActionType.FINISH:
                self._handle_finish(action)
                break

        # 关闭 LLM 和 XPU 连接，但保留容器
        self._close_clients()
        # 清理快照镜像释放磁盘，验证阶段不需要回滚
        self._env.cleanup_snapshots()

        if not self._state.completed:
            logger.warning("达到最大迭代次数，任务未完成")

        return SetupResult(
            repo_url=self._state.repo_url,
            container_id=container_id,
            completed=self._state.completed,
            steps_taken=self._state.step,
            final_message=self._state.final_message or "达到最大迭代次数，任务未完成",
            history=self._state.history,
            last_verify_messages=self._last_verify_messages,
        )

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
            # 区分"已用过"和"当前无此建议"，给出更明确的指导
            sid = action.xpu_suggestion_id
            if sid in self._state.failed_suggestions:
                msg = (
                    f"[XPU BLOCKED] 建议 {sid} 本轮已使用过，不能重复执行。"
                    "请改用 SHELL_COMMAND 自行实现类似逻辑，或尝试其他思路。"
                )
            else:
                msg = (
                    f"[XPU BLOCKED] 建议 {sid} 当前不可用（可能当前无错误上下文，或该 ID 不存在）。"
                    "请改用 SHELL_COMMAND。"
                )
            logger.warning(f"未找到 XPU 建议: {sid}，原因: {msg[:60]}")
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": msg,
                    "stderr": "",
                },
            })
            return

        # 记录执行前的错误
        error_before = self._state.last_error or ""

        # A. 存档 (Checkpoint)
        ckpt_tag = f"step_{self._state.step}_pre_xpu"
        self._env.create_checkpoint(ckpt_tag)
        logger.info(f"创建快照 {ckpt_tag}，开始推测执行 XPU 建议")

        # B. 试错 (Trial)
        if not suggestion.commands:
            logger.warning(f"XPU 建议 {suggestion.id} 的 commands 为空，跳过执行")
            self._state.record_failed_suggestion(suggestion.id)
            self._state.add_to_history({
                "action": action.to_dict(),
                "result": {
                    "exit_code": 1,
                    "stdout": f"[XPU SKIP] {suggestion.id}：commands 为空，未执行",
                    "stderr": "",
                },
            })
            return

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
            self._state.last_error = error_before  # 恢复原错误
        else:
            logger.info(f"XPU 建议 {suggestion.id} 验证通过")
            self._state.last_error = None
        # 无论成功失败，都标记为"已尝试"，防止 LLM 对同一条建议无限循环
        self._state.record_failed_suggestion(suggestion.id)

        # 记录历史：必须含 "result" key，否则 llm_engine 不会生成 user 观察消息
        cmd_outputs = "\n".join(
            f"$ {log.get('command', '')}\n{(log.get('stdout') or log.get('stderr') or '')[:300]}"
            for log in [l.to_dict() for l in logs[:3]]
        )
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[XPU {outcome}] {suggestion.id}\n{cmd_outputs}",
                "stderr": "",
            },
        })

    def _handle_set_env(self, action: AgentAction) -> None:
        """处理 SET_ENV 动作"""
        if not action.env_key or action.env_value is None:
            logger.warning("SET_ENV 缺少 env_key 或 env_value")
            return

        self._env.set_env(action.env_key, action.env_value)
        self._state.env_vars[action.env_key] = action.env_value
        # 清除 last_error，避免 LLM 下一步仍看到旧报错而重复 SET_ENV
        self._state.last_error = None

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[SET_ENV] {action.env_key}={action.env_value}，变量已生效，请继续执行后续命令。",
                "stderr": "",
            },
        })

    def _handle_rollback_env(self, action: AgentAction) -> None:
        """处理 ROLLBACK_ENV 动作：回滚容器到最近快照"""
        try:
            success = self._env.rollback_to_checkpoint()
            status = "成功" if success else "失败（无可用快照）"
        except Exception as e:
            logger.error(f"[ROLLBACK_ENV] 基础设施异常: {e}")
            success = False
            status = f"失败（基础设施异常: {e}）"
        logger.info(f"[ROLLBACK_ENV] 回滚 {status}")
        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0 if success else 1,
                "stdout": f"[ROLLBACK_ENV] {status}",
                "stderr": "",
            },
        })

    def _handle_verify(self, action: AgentAction) -> bool:
        """处理 VERIFY 动作：调用 VerifierAgent 运行 pytest，把结果反馈给 LLM。

        返回 True 表示验证通过（调用方应退出主循环），
        返回 False 表示验证失败（调用方继续循环，让 LLM 根据 pytest 输出继续修复）。
        """
        logger.info("[VERIFY] 开始 pytest 验证")
        hint = action.verify_hint or ""
        if hint:
            logger.info(f"[VERIFY] 传递 hint 给 Verifier: {hint}")
        verifier = VerifierAgent(self._env, hint=hint)
        result = verifier.verify()

        logger.info(
            f"[VERIFY] 结果: success={result.success}, "
            f"framework={result.test_framework}, "
            f"collected={result.collect_count}, exit_code={result.exit_code}"
        )

        # 组织反馈，无论成功失败都写入历史供 LLM 参考
        verify_summary = (
            f"验证框架: {result.test_framework}\n"
            f"收集测试数: {result.collect_count}\n"
            f"退出码: {result.exit_code}\n"
            f"输出:\n{result.stdout}\n"
            f"错误:\n{result.stderr}"
        )

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": result.exit_code,
                "stdout": (
                    f"[VERIFY] success={result.success}, framework={result.test_framework}, "
                    f"collected={result.collect_count}\n{result.stdout or ''}"
                )[:1000],
                "stderr": (result.stderr or "")[:500],
            },
        })

        if result.success:
            logger.info("[VERIFY] 验证通过，自动标记任务完成")
            self._state.completed = True
            self._state.final_message = f"pytest 验证通过，{result.collect_count} 个测试用例"
            self._state.last_error = None
            self._last_verify_messages = result.messages
            return True
        else:
            logger.warning("[VERIFY] 验证失败，将 pytest 输出反馈给 LLM 继续修复")
            self._state.last_error = verify_summary
            return False

    def _handle_finish(self, action: AgentAction) -> None:
        """处理 FINISH 动作"""
        self._state.completed = True
        self._state.final_message = action.message or "任务完成"

        self._state.add_to_history({
            "action": action.to_dict(),
            "result": {
                "exit_code": 0,
                "stdout": f"[FINISH] {self._state.final_message}",
                "stderr": "",
            },
        })

    def _close_clients(self) -> None:
        """关闭 LLM 和 XPU 连接（不销毁容器）"""
        logger.info("关闭 LLM/XPU 连接...")
        self._llm.close()
        if hasattr(self._xpu, "close"):
            self._xpu.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close_clients()
        self._env.cleanup()
        return False
