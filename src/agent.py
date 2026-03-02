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

            elif action.action_type == ActionType.ROLLBACK_ENV:
                self._handle_rollback_env(action)

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

        # B. LLM 适配命令：根据建议思路 + 当前上下文生成具体命令
        cwd = self._env.exec_run("pwd").stdout.strip()
        os_info = self._env.exec_run("cat /etc/os-release | head -2").stdout.strip()

        advice_nl = [line for line in suggestion.description.split("\n") if line.strip()]
        adapted_commands = self._llm.adapt_xpu_commands(
            advice_nl=advice_nl,
            last_error=error_before,
            cwd=cwd,
            os_info=os_info,
        )

        if not adapted_commands:
            logger.warning(f"LLM 未能生成适配命令，回滚并跳过建议 {suggestion.id}")
            self._env.rollback_to_checkpoint()
            self._state.record_failed_suggestion(suggestion.id)
            self._state.last_error = error_before
            self._state.add_to_history({
                "action": action.to_dict(),
                "xpu_suggestion": suggestion.to_dict(),
                "outcome": "SKIP",
                "reason": "LLM 适配命令为空",
            })
            return

        logger.info(f"LLM 适配后命令 ({len(adapted_commands)} 条): {adapted_commands}")

        # C. 试错 (Trial) — 执行 LLM 适配后的命令
        success = True
        logs: list[CommandResult] = []
        for cmd in adapted_commands:
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

        # 记录历史（包含原始建议和 LLM 适配后的命令）
        self._state.add_to_history({
            "action": action.to_dict(),
            "xpu_suggestion": suggestion.to_dict(),
            "adapted_commands": adapted_commands,
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

    def _handle_rollback_env(self, action: AgentAction) -> None:
        """处理 ROLLBACK_ENV 动作：回滚容器到最近快照"""
        success = self._env.rollback_to_checkpoint()
        status = "成功" if success else "失败（无可用快照）"
        logger.info(f"[ROLLBACK_ENV] 回滚 {status}")
        self._state.add_to_history({
            "action": action.to_dict(),
            "rollback_status": status,
        })

    def _handle_finish(self, action: AgentAction) -> None:
        """处理 FINISH 动作"""
        self._state.completed = True
        self._state.final_message = action.message or "任务完成"

        self._state.add_to_history({
            "action": action.to_dict(),
        })

        # 任务成功后尝试存储经验到向量数据库
        self._store_experience_if_applicable()

    def _store_experience_if_applicable(self) -> None:
        """任务成功完成后，将本次修复经验存入 XPU 向量数据库（如果可用）
        复用 online_xpu_extractor.py 的完整 LLM 提取管道。
        """
        from .xpu_client import VectorXPUClient
        if not isinstance(self._xpu, VectorXPUClient):
            return
        if not self._state.completed:
            return

        try:
            import json
            import shutil
            import tempfile
            import sys
            import os
            from pathlib import Path

            _src_dir = os.path.dirname(os.path.abspath(__file__))
            if _src_dir not in sys.path:
                sys.path.insert(0, _src_dir)

            # Step 1: 将 agent history 转换为 Repo2Run 兼容的轨迹 JSONL 格式
            traj = []
            for entry in self._state.history:
                action = entry.get("action", {})
                result = entry.get("result", {})

                # assistant 消息：将命令包装为 bash 代码块
                cmd = action.get("content", {}).get("command")
                if cmd:
                    traj.append({
                        "role": "assistant",
                        "content": f"执行命令:\n```bash\n{cmd}\n```"
                    })

                # system 消息：命令执行输出/错误
                if result:
                    output = result.get("stderr") or result.get("stdout") or ""
                    if output:
                        traj.append({
                            "role": "system",
                            "content": output
                        })

            if not traj:
                logger.debug("[XPU Store] 轨迹为空，跳过经验存储")
                return

            # Step 2: 写入临时 JSONL 文件（Repo2Run 命名格式: {safe_name}@HEAD.jsonl）
            tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_agent_"))
            try:
                repo_path = self._state.repo_url.rstrip("/")
                if "github.com/" in repo_path:
                    repo_path = repo_path.split("github.com/")[-1]
                safe_name = repo_path.replace("/", "__")

                traj_dir = tmp_dir / "trajs"
                traj_dir.mkdir()
                jsonl_path = traj_dir / f"{safe_name}@HEAD.jsonl"

                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for step in traj:
                        f.write(json.dumps(step, ensure_ascii=False) + "\n")

                # Step 3: LLM 提取（复用 extract_xpu_from_trajs_mvp）
                extracted_file = tmp_dir / "extracted.jsonl"
                from xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
                extract_xpu_from_trajs(jsonl_path, extracted_file)

                # Step 4: 收集所有有效经验（一条轨迹可能提炼出多条 XPU）
                xpu_objects = []
                if extracted_file.exists():
                    with open(extracted_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            rec = json.loads(line)
                            if rec.get("llm_decision") == "xpu" and rec.get("xpu"):
                                xpu_objects.append(rec["xpu"])

                if not xpu_objects:
                    logger.debug("[XPU Store] LLM 决定跳过，无有效经验存储")
                    return

                logger.info(f"[XPU Store] LLM 提取出 {len(xpu_objects)} 条经验，逐条入库")

                # Step 5 & 6: 逐条构建 XpuEntry 并去重入库
                from xpu.xpu_adapter import XpuEntry, XpuAtom
                from xpu.xpu_vector_store import build_xpu_text, text_to_embedding
                from xpu.xpu_dedup import dedup_and_store

                for i, xpu_obj in enumerate(xpu_objects):
                    atoms = [XpuAtom(name=a.get("name", ""), args=a.get("args", {}))
                             for a in xpu_obj.get("atoms", [])]
                    xpu_entry = XpuEntry(
                        id=xpu_obj.get("id"),
                        context=xpu_obj.get("context", {}),
                        signals=xpu_obj.get("signals", {}),
                        advice_nl=xpu_obj.get("advice_nl", []),
                        atoms=atoms,
                    )

                    text = build_xpu_text(xpu_entry)
                    embedding = text_to_embedding(text)
                    dedup_result = dedup_and_store(self._xpu._store, xpu_entry, embedding, use_llm=True)
                    logger.info(f"[XPU Store] [{i+1}/{len(xpu_objects)}] {dedup_result['action']}: {dedup_result['reason']}")

            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"[XPU Store] 经验存储失败（不影响任务结果）: {e}")

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
