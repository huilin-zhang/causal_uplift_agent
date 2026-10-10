"""Keep README workflow and role descriptions identical to the canonical overview."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "<!-- growth-overview:start -->"
END = "<!-- growth-overview:end -->"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = (ROOT / "docs/manual/Growth_Marketing_Agents_Overview.md").read_text(encoding="utf-8")
    block = source[source.index("## Workflow"):].strip()
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    before, remaining = text.split(START, 1)
    current, after = remaining.split(END, 1)
    expected = "\n\n" + block + "\n\n"
    if args.check:
        if current != expected:
            raise SystemExit("README workflow/overview differs; run python docs/tools/sync_agent_overview.py")
        print("README workflow and Agent Overview match the canonical document")
    else:
        path.write_text(before + START + expected + END + after, encoding="utf-8")


if __name__ == "__main__":
    main()
