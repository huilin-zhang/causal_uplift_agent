"""Roll the served model back to the previous champion.

    python -m scripts.rollback_model            # swap champion and previous_champion
    python -m scripts.rollback_model --status   # show which versions hold the aliases

After a rollback, call POST /model/reload (or restart the API) so serving
picks up the new champion. Code guide: section "MLflow registry" (sec:registry).
"""

from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.services.registry import dump, registry_status, rollback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    cfg = get_settings()
    print(dump(registry_status(cfg) if args.status else rollback(cfg)))


if __name__ == "__main__":
    main()
