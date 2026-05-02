#!/usr/bin/env python3
"""
Multi-Domain Support Triage Agent
Entry point for the HackerRank Orchestrate Hackathon.

Usage:
    python main.py                          # Process support_tickets.csv
    python main.py --input path/to/csv     # Custom input
    python main.py --dry-run               # First 3 rows only
    python main.py --verbose               # Verbose logging
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.triage_agent import TriageAgent
from utils.config import Config
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV (default: ../support_tickets/support_tickets.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output CSV (default: ../support_tickets/output.csv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first 3 rows for testing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the vector index",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("triage_agent", log_level)

    logger.info("=" * 60)
    logger.info("  Multi-Domain Support Triage Agent")
    logger.info("  HackerRank Orchestrate Hackathon 2026")
    logger.info("=" * 60)

    # Load config
    config = Config()

    # Resolve paths
    base_dir = Path(__file__).parent.parent
    input_path = args.input or str(base_dir / "support_tickets" / "support_tickets.csv")
    output_path = args.output or str(base_dir / "support_tickets" / "output.csv")

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Initialize and run agent
    try:
        agent = TriageAgent(
            config=config,
            rebuild_index=args.rebuild_index,
            verbose=args.verbose,
        )

        agent.process_csv(
            input_path=input_path,
            output_path=output_path,
            dry_run=args.dry_run,
        )

        logger.info("✅ Processing complete!")
        logger.info(f"Results written to: {output_path}")

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
