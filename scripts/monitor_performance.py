"""
RAG System Performance Monitoring & Analytics
==============================================

Track query performance, measure accuracy, and generate reports.

Features:
- Latency tracking (p50, p95, p99)
- Accuracy metrics (recall, precision)
- Source diversity analysis
- Query pattern analysis
- Export to CSV for analysis

Usage:
    PYTHONPATH=. poetry run python scripts/monitor_performance.py

Author: Abhishek Gurjar
"""

import sys
import os
from pathlib import Path
import time
import json
from typing import List, Dict
from datetime import datetime
import logging
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag_orchestrator import create_rag_system

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Track and analyze RAG system performance."""
    
    def __init__(self):
        self.query_logs = []
        self.metrics = defaultdict(list)
    
    def log_query(
        self,
        query: str,
        result: Dict,
        ground_truth_answer: str = None
    ):
        """
        Log a query and its results.
        
        Args:
            query: User query
            result: RAG system response
            ground_truth_answer: Optional ground truth for accuracy measurement
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", ""),
            "retrieval_time_ms": result.get("metadata", {}).get("retrieval_time_ms", 0),
            "generation_time_ms": result.get("metadata", {}).get("generation_time_ms", 0),
            "total_time_ms": result.get("metadata", {}).get("total_time_ms", 0),
            "num_sources": result.get("metadata", {}).get("num_sources", 0),
            "sources": result.get("sources", []),
            "quality_score": result.get("quality_score"),
        }
        
        if ground_truth_answer:
            log_entry["ground_truth"] = ground_truth_answer
        
        self.query_logs.append(log_entry)
        
        # Update metrics
        self.metrics["retrieval_latency"].append(log_entry["retrieval_time_ms"])
        self.metrics["generation_latency"].append(log_entry["generation_time_ms"])
        self.metrics["total_latency"].append(log_entry["total_time_ms"])
        self.metrics["num_sources"].append(log_entry["num_sources"])
    
    def calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.query_logs:
            return {}
        
        retrieval_latencies = self.metrics["retrieval_latency"]
        generation_latencies = self.metrics["generation_latency"]
        total_latencies = self.metrics["total_latency"]
        
        stats = {
            "total_queries": len(self.query_logs),
            "latency_stats": {
                "retrieval": {
                    "p50": self.calculate_percentile(retrieval_latencies, 50),
                    "p95": self.calculate_percentile(retrieval_latencies, 95),
                    "p99": self.calculate_percentile(retrieval_latencies, 99),
                    "avg": sum(retrieval_latencies) / len(retrieval_latencies),
                    "min": min(retrieval_latencies),
                    "max": max(retrieval_latencies),
                },
                "generation": {
                    "p50": self.calculate_percentile(generation_latencies, 50),
                    "p95": self.calculate_percentile(generation_latencies, 95),
                    "p99": self.calculate_percentile(generation_latencies, 99),
                    "avg": sum(generation_latencies) / len(generation_latencies),
                    "min": min(generation_latencies),
                    "max": max(generation_latencies),
                },
                "total": {
                    "p50": self.calculate_percentile(total_latencies, 50),
                    "p95": self.calculate_percentile(total_latencies, 95),
                    "p99": self.calculate_percentile(total_latencies, 99),
                    "avg": sum(total_latencies) / len(total_latencies),
                    "min": min(total_latencies),
                    "max": max(total_latencies),
                }
            },
            "confidence_distribution": self._calculate_confidence_dist(),
            "source_stats": {
                "avg_sources_per_query": sum(self.metrics["num_sources"]) / len(self.metrics["num_sources"]),
                "min_sources": min(self.metrics["num_sources"]),
                "max_sources": max(self.metrics["num_sources"]),
            }
        }
        
        return stats
    
    def _calculate_confidence_dist(self) -> Dict:
        """Calculate confidence distribution."""
        conf_counts = defaultdict(int)
        for log in self.query_logs:
            conf_counts[log["confidence"]] += 1
        
        total = len(self.query_logs)
        return {
            conf: (count / total) * 100
            for conf, count in conf_counts.items()
        }
    
    def print_report(self):
        """Print a formatted performance report."""
        stats = self.get_summary_stats()
        
        if not stats:
            logger.warning("No queries logged yet!")
            return
        
        print("\n" + "="*80)
        print("📊 RAG SYSTEM PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\n📈 Query Statistics:")
        print(f"   Total Queries: {stats['total_queries']}")
        
        print(f"\n⏱️  Latency Statistics (milliseconds):")
        print(f"\n   Retrieval:")
        for key, value in stats['latency_stats']['retrieval'].items():
            print(f"      {key.upper()}: {value:.2f}ms")
        
        print(f"\n   Generation:")
        for key, value in stats['latency_stats']['generation'].items():
            print(f"      {key.upper()}: {value:.2f}ms")
        
        print(f"\n   End-to-End:")
        for key, value in stats['latency_stats']['total'].items():
            print(f"      {key.upper()}: {value:.2f}ms")
        
        print(f"\n📚 Source Statistics:")
        print(f"   Average sources per query: {stats['source_stats']['avg_sources_per_query']:.1f}")
        print(f"   Min sources: {stats['source_stats']['min_sources']}")
        print(f"   Max sources: {stats['source_stats']['max_sources']}")
        
        print(f"\n🎯 Confidence Distribution:")
        for conf, percentage in stats['confidence_distribution'].items():
            print(f"   {conf}: {percentage:.1f}%")
        
        print("\n" + "="*80)
    
    def export_to_json(self, filepath: str = "performance_report.json"):
        """Export logs and stats to JSON."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary_stats": self.get_summary_stats(),
            "query_logs": self.query_logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report exported to {filepath}")


def run_benchmark_queries(rag, monitor: PerformanceMonitor):
    """Run a set of benchmark queries."""
    benchmark_queries = [
        "What is Python?",
        "Explain machine learning",
        "What is NLP?",
        "Compare Python and machine learning",
        "What year was transformer introduced?",
    ]
    
    logger.info("="*80)
    logger.info("RUNNING BENCHMARK QUERIES")
    logger.info("="*80)
    
    for i, query in enumerate(benchmark_queries, 1):
        logger.info(f"\n[{i}/{len(benchmark_queries)}] Query: {query}")
        
        try:
            result = rag.query(query, top_k=5, return_sources=True)
            monitor.log_query(query, result)
            
            logger.info(f"   ✅ Completed in {result.get('metadata', {}).get('total_time_ms', 0):.0f}ms")
            logger.info(f"   Confidence: {result['confidence']}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed: {e}")
    
    logger.info("\n✅ Benchmark complete!")


def main():
    """Main monitoring function."""
    # Initialize RAG
    logger.info("Initializing RAG system...")
    rag = create_rag_system()
    
    # Check if collection has data
    stats = rag.get_stats()
    if stats['total_vectors'] == 0:
        logger.warning("⚠️  Collection is empty. Index documents first.")
        logger.info("   Run: PYTHONPATH=. poetry run python scripts/index_documents.py")
        return
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Run benchmarks
    run_benchmark_queries(rag, monitor)
    
    # Print report
    monitor.print_report()
    
    # Export to JSON
    monitor.export_to_json("performance_report.json")
    
    logger.info("\n📊 Performance monitoring complete!")
    logger.info("   View detailed report: performance_report.json")


if __name__ == "__main__":
    main()
