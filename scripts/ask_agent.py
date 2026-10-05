"""Ask the agent a question from the command line.

    python -m scripts.ask_agent "Plan this month's retention campaign." --budget 500
    python -m scripts.ask_agent "Should we send an offer to customer 42?"

Code guide: section "Agent graph" (sec:graph).
"""

from __future__ import annotations

import argparse

from app.agents.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--trace", action="store_true", help="print which sub-agents ran")
    args = parser.parse_args()
    state = run_agent(args.question, args.budget)
    print(state["narrative"])
    if args.trace:
        print("\nTrace:")
        for line in state["trace"]:
            print("  " + line)


if __name__ == "__main__":
    main()
