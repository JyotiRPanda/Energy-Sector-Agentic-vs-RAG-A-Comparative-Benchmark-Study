from __future__ import annotations

import argparse
import json

from gri_benchmark.runner import run_from_config
from gri_benchmark.settings import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG vs agentic benchmark")
    parser.add_argument("--config", required=True, help="Path to benchmark YAML config")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Optional local env file for future private Foundry settings",
    )
    parser.add_argument(
        "--use-domain-aware-retrieval",
        action="store_true",
        help="Enable domain-aware retrieval (validated improvement from STEP 1-4)",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    summary = run_from_config(args.config, use_domain_aware_retrieval=args.use_domain_aware_retrieval)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
