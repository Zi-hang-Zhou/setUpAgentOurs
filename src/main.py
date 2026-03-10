"""
命令行入口 — 三阶段流程编排
阶段1: Setup（Agent 推理，最多50步）
阶段2: Phase 2 诉讼裁决（仅 Setup 主动 FINISH 时运行）
阶段3: Report（结果输出）
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from .logger import get_logger
from .agent import SpeculativeSetupAgent
from .prosecutor_agent import ProsecutorAgent
from .judge_agent import JudgeAgent
from .models import ProsecutionResult

logger = get_logger("main")


def _build_traj_from_history(history: list[dict]) -> list[dict]:
    """将 agent history 转为 JSONL 格式
    assistant 消息包装 bash 命令；system 消息包装 stderr/stdout
    """
    traj = []
    for entry in history:
        action = entry.get("action", {})
        result = entry.get("result", {})

        cmd = action.get("content", {}).get("command")
        if cmd:
            traj.append({
                "role": "assistant",
                "content": f"执行命令:\n```bash\n{cmd}\n```"
            })

        if result:
            output = result.get("stderr") or result.get("stdout") or ""
            if output:
                traj.append({
                    "role": "system",
                    "content": output
                })
    return traj


def _store_xpu_experience(
    xpu_client: Any,
    setup_result: Any,
    prosecution: "ProsecutionResult | None",
    judgment: "dict | None",
) -> None:
    """Phase 2 结束后统一触发 XPU 经验提取与入库"""
    from .xpu_client import VectorXPUClient
    if not isinstance(xpu_client, VectorXPUClient):
        return
    if not setup_result.completed:
        return

    try:
        traj = _build_traj_from_history(setup_result.history)
        if not traj:
            logger.debug("[XPU Store] 轨迹为空，跳过经验存储")
            return

        # 构造 phase2_context
        phase2_context = None
        if prosecution is not None:
            verifier_msgs = setup_result.last_verify_messages or []
            verifier_text = ""
            for m in verifier_msgs:
                verifier_text += str(m.get("content", ""))
            phase2_context = {
                "prosecution_charges": prosecution.charges,
                "verdict": judgment["verdict"] if judgment else None,
                "judge_reasoning": judgment["reasoning"] if judgment else "",
                "verifier_summary": verifier_text[:500],
            }

        tmp_dir = Path(tempfile.mkdtemp(prefix="xpu_agent_"))
        try:
            repo_path = setup_result.repo_url.rstrip("/")
            if "github.com/" in repo_path:
                repo_path = repo_path.split("github.com/")[-1]
            safe_name = repo_path.replace("/", "__")

            traj_dir = tmp_dir / "trajs"
            traj_dir.mkdir()
            jsonl_path = traj_dir / f"{safe_name}@HEAD.jsonl"

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for step in traj:
                    f.write(json.dumps(step, ensure_ascii=False) + "\n")

            extracted_file = tmp_dir / "extracted.jsonl"
            from .xpu.extract_xpu_from_trajs_mvp import extract_xpu_from_trajs
            extract_xpu_from_trajs(jsonl_path, extracted_file, phase2_context=phase2_context)

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

            from .xpu.xpu_adapter import XpuEntry, XpuAtom
            from .xpu.xpu_vector_store import build_xpu_text, text_to_embedding
            from .xpu.xpu_dedup import dedup_and_store

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
                dedup_result = dedup_and_store(xpu_client._store, xpu_entry, embedding, use_llm=True)
                logger.info(f"[XPU Store] [{i+1}/{len(xpu_objects)}] {dedup_result['action']}: {dedup_result['reason']}")

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except Exception as e:
        logger.warning(f"[XPU Store] 经验存储失败（不影响任务结果）: {e}")


def main() -> int:
    """主入口函数"""
    if len(sys.argv) < 2:
        print("用法: python -m src.main <git_repo_url> [max_iterations]", file=sys.stderr)
        print("示例: python -m src.main https://github.com/user/repo", file=sys.stderr)
        return 1

    repo_url = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    logger.info("启动三阶段流程")
    logger.info(f"目标仓库: {repo_url}")
    logger.info(f"最大迭代次数: {max_iterations}")

    # ── 阶段1: Setup ──
    logger.info("=== 阶段1: Setup（Agent 推理）===")
    agent = SpeculativeSetupAgent(repo_url, max_iterations)
    setup_result = agent.run()
    logger.info(
        f"Setup 完成: completed={setup_result.completed}, "
        f"steps={setup_result.steps_taken}, container={setup_result.container_id[:12]}"
    )

    # ── 阶段2: Phase 2 诉讼裁决 ──
    logger.info("=== 阶段2: Phase 2 诉讼裁决 ===")

    phase2_success: bool
    phase2_reason: str
    prosecution_dict = None
    prosecution = None
    judgment = None

    if setup_result.completed:
        logger.info("Setup Agent 主动 FINISH，启动 Phase 2 诉讼模型")

        # 检察官调查
        logger.info("--- 检察官调查阶段 ---")
        prosecutor = ProsecutorAgent(
            agent.env,
            setup_result.history,
            setup_result.last_verify_messages,
        )
        prosecution = prosecutor.investigate()
        prosecution_dict = {
            "prosecute": prosecution.prosecute,
            "charges": prosecution.charges,
        }
        logger.info(f"检察官调查完成: prosecute={prosecution.prosecute}, 指控数={len(prosecution.charges)}")

        if not prosecution.prosecute:
            phase2_success = True
            phase2_reason = "Prosecutor 未发现实质问题"
            logger.info("检察官选择不起诉，直接判定 success=True")
        else:
            # 法官裁决
            logger.info("--- 法官裁决阶段 ---")
            judgment = JudgeAgent(
                setup_result.history,
                setup_result.last_verify_messages,
                prosecution,
            ).rule()
            phase2_success = (judgment["verdict"] == "not_guilty")
            phase2_reason = judgment["reasoning"]
            logger.info(
                f"法官裁决: verdict={judgment['verdict']}, "
                f"reasoning={phase2_reason[:100]}"
            )
    else:
        phase2_success = False
        phase2_reason = f"Setup Agent 超时（{setup_result.steps_taken} 步），未主动 FINISH"
        logger.info(f"Setup 未完成，跳过 Phase 2: {phase2_reason}")

    # ── XPU 经验提取（所有信号就绪后统一触发）──
    logger.info("=== XPU 经验提取 ===")
    _store_xpu_experience(
        xpu_client=agent._xpu,
        setup_result=setup_result,
        prosecution=prosecution,
        judgment=judgment,
    )

    # ── 清理容器 ──
    agent.env.destroy()
    logger.info("容器已销毁")

    # ── 阶段3: Report ──
    logger.info("=== 阶段3: Report（结果输出）===")
    report = {
        "repo_url": repo_url,
        "setup": setup_result.to_dict(),
        "phase2": {
            "success": phase2_success,
            "reason": phase2_reason,
            "prosecution": prosecution_dict,
        },
    }

    log_dir = Path("log")
    log_dir.mkdir(exist_ok=True)
    safe_name = repo_url.rstrip("/").split("/")[-1]
    output_path = log_dir / f"{safe_name}_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"结果已写入: {output_path}")

    # 屏幕输出摘要
    print(f"\n{'='*50}")
    print(f"仓库: {repo_url}")
    print(f"Setup: {'完成' if setup_result.completed else '未完成'} ({setup_result.steps_taken} 步)")
    print(f"Phase2: {'通过' if phase2_success else '失败'}")
    print(f"裁决原因: {phase2_reason}")
    print(f"详细结果: {output_path}")
    print(f"{'='*50}")

    return 0 if phase2_success else 1


if __name__ == "__main__":
    sys.exit(main())
