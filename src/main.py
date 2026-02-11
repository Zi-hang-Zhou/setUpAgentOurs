"""
命令行入口
"""

import sys

from .logger import get_logger
from .agent import SpeculativeSetupAgent

logger = get_logger("main")


def main() -> int:
    """主入口函数"""
    if len(sys.argv) < 2:
        print("用法: python -m src.main <git_repo_url> [max_iterations]", file=sys.stderr)
        print("示例: python -m src.main https://github.com/user/repo", file=sys.stderr)
        return 1

    repo_url = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    logger.info(f"启动 Speculative Setup Agent")
    logger.info(f"目标仓库: {repo_url}")
    logger.info(f"最大迭代次数: {max_iterations}")

    with SpeculativeSetupAgent(repo_url, max_iterations) as agent:
        result = agent.run()

    print(f"\n{'='*50}")
    print(f"最终结果: {result}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
