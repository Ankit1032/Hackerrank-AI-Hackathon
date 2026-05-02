"""
TriageAgent — top-level orchestrator.

Responsibilities:
1. Load and index the corpus (once, cached to disk)
2. Accept a CSV of tickets
3. Run each ticket through the LangGraph pipeline
4. Write results to output CSV
5. Print progress & summary stats
"""

import csv
import logging
import time
from pathlib import Path
from typing import List, Optional

from utils.config import Config
from utils.models import SupportTicket, TriageResult
from utils.logger import get_logger
from data_loader.corpus_loader import load_corpus, chunk_documents
from retriever.vector_retriever import VectorRetriever
from agents.triage_graph import TriageGraph

logger = get_logger("triage_agent")

OUTPUT_COLUMNS = [
    "issue", "subject", "company",
    "status", "product_area", "response", "justification", "request_type",
]


class TriageAgent:
    """
    High-level orchestrator for the multi-domain support triage system.
    """

    def __init__(self, config: Config, rebuild_index: bool = False, verbose: bool = False):
        config.validate()
        self.config = config
        self.verbose = verbose

        logger.info("Initializing TriageAgent...")

        # Step 1: Initialize retriever (loads or builds index)
        self.retriever = VectorRetriever(config=config, rebuild=rebuild_index)

        # Step 2: Build corpus index if not already loaded
        if self.retriever.embeddings is None or len(self.retriever.chunks) == 0:
            self._build_corpus_index()

        # Step 3: Initialize LangGraph triage pipeline
        self.graph = TriageGraph(config=config, retriever=self.retriever)

        logger.info("TriageAgent ready.")

    def _build_corpus_index(self):
        """Load corpus from disk and build vector index."""
        logger.info("Building corpus index...")

        raw_docs = load_corpus(self.config.corpus_dirs)

        if not raw_docs:
            logger.warning(
                "No corpus documents found. "
                "Ensure data/hackerrank/, data/claude/, data/visa/ directories exist."
            )
            return

        chunks = chunk_documents(
            raw_docs,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        self.retriever.build_index(chunks)

    def process_csv(
        self,
        input_path: str,
        output_path: str,
        dry_run: bool = False,
    ):
        """
        Process all tickets in the input CSV and write results to output CSV.
        """
        tickets = self._read_tickets(input_path)

        if dry_run:
            tickets = tickets[:3]
            logger.info(f"DRY RUN: processing first {len(tickets)} tickets only")

        logger.info(f"Processing {len(tickets)} tickets...")

        results = []
        stats = {
            "replied": 0,
            "escalated": 0,
            "product_issue": 0,
            "feature_request": 0,
            "bug": 0,
            "invalid": 0,
        }

        for i, ticket in enumerate(tickets):
            start = time.time()
            logger.info(
                f"[{i+1}/{len(tickets)}] Processing: {ticket.subject[:50] or ticket.issue[:50]}..."
            )

            try:
                result = self.graph.run(ticket)
                results.append((ticket, result))

                elapsed = time.time() - start
                logger.info(
                    f"  → status={result.status} | area={result.product_area} "
                    f"| type={result.request_type} ({elapsed:.1f}s)"
                )

                stats[result.status] = stats.get(result.status, 0) + 1
                stats[result.request_type] = stats.get(result.request_type, 0) + 1

            except Exception as e:
                logger.error(f"  ✗ Failed: {e}", exc_info=self.verbose)
                # Fail-safe: escalate on error
                fallback = TriageResult(
                    status="escalated",
                    product_area="general_support",
                    response=(
                        "We encountered an issue processing your request. "
                        "A human agent will review it shortly."
                    ),
                    justification=f"Processing error — escalated as a precaution: {str(e)[:100]}",
                    request_type="product_issue",
                )
                results.append((ticket, fallback))
                stats["escalated"] = stats.get("escalated", 0) + 1

        self._write_output(output_path, results)
        self._print_summary(stats, len(tickets))

    def _read_tickets(self, path: str) -> List[SupportTicket]:
        """Read and parse the input CSV."""
        tickets = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Handle varied column name casings
                issue = row.get("Issue") or row.get("issue") or ""
                subject = row.get("Subject") or row.get("subject") or ""
                company = row.get("Company") or row.get("company") or "None"

                tickets.append(
                    SupportTicket(
                        issue=issue.strip(),
                        subject=subject.strip(),
                        company=company.strip(),
                        row_index=i,
                    )
                )

        logger.info(f"Read {len(tickets)} tickets from {path}")
        return tickets

    def _write_output(self, path: str, results: List[tuple]):
        """Write results to the output CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for ticket, result in results:
                writer.writerow(result.to_dict(ticket))

        logger.info(f"Results written to: {path}")

    def _print_summary(self, stats: dict, total: int):
        """Print a processing summary."""
        logger.info("\n" + "=" * 50)
        logger.info("  PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Total tickets  : {total}")
        logger.info(f"  Replied        : {stats.get('replied', 0)}")
        logger.info(f"  Escalated      : {stats.get('escalated', 0)}")
        logger.info(f"  product_issue  : {stats.get('product_issue', 0)}")
        logger.info(f"  feature_request: {stats.get('feature_request', 0)}")
        logger.info(f"  bug            : {stats.get('bug', 0)}")
        logger.info(f"  invalid        : {stats.get('invalid', 0)}")
        logger.info("=" * 50 + "\n")
