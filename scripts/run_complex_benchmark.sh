#!/usr/bin/env bash
# Quick reference for running complex subset benchmark

set -e

# Configuration
CONFIG="configs/benchmark_complex_subset.yaml"
ENV_FILE=".env"
OUTPUT_DIR="results/complex"
REPORT_PATH="docs/generated/complex_subset_report.md"

echo "================================================"
echo "Complex-Only Benchmark Runner"
echo "================================================"
echo ""

# Check prerequisites
if [ ! -f "$ENV_FILE" ]; then
    echo " Error: $ENV_FILE not found"
    echo "   Please create .env with Azure OpenAI credentials"
    echo "   Required: PROJECT_ENDPOINT, API_KEY, MODEL_DEPLOYMENT, EMBEDDING_DEPLOYMENT"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo " Error: $CONFIG not found"
    exit 1
fi

echo " Configuration found: $CONFIG"
echo " Environment file: $ENV_FILE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$REPORT_PATH")"

echo "Step 1: Building retrieval corpus..."
python scripts/build_retrieval_corpus.py --config "$CONFIG" || {
    echo "  Corpus building failed (may already exist)"
}
echo ""

echo "Step 2: Running benchmark..."
python scripts/run_complex_subset_benchmark.py \
    --config "$CONFIG" \
    --env-file "$ENV_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --report-md "$REPORT_PATH"
echo ""

echo " Benchmark complete!"
echo ""
echo "Outputs:"
echo "  - Summary:       $OUTPUT_DIR/live_summary.json"
echo "  - Predictions:   $OUTPUT_DIR/live_predictions.json"
echo "  - RAG Full:      $OUTPUT_DIR/traditional_rag_live_predictions.json"
echo "  - Agentic Full:  $OUTPUT_DIR/agentic_multi_tool_live_predictions.json"
echo "  - Report:        $REPORT_PATH"
echo ""

# Display key metrics
echo "Quick Summary:"
python << 'EOF'
import json
from pathlib import Path

summary_path = Path("results/complex/live_summary.json")
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    
    rag = summary.get("traditional_rag", {})
    agentic = summary.get("agentic_multi_tool", {})
    
    print(f"  Traditional RAG:")
    print(f"    - Exact Match: {rag.get('exact_match', 0):.4f}")
    print(f"    - Latency: {rag.get('latency_ms', 0):.1f} ms")
    print(f"    - Cost: ${rag.get('total_cost_usd', 0):.4f}")
    print()
    print(f"  Agentic Multi-Tool:")
    print(f"    - Exact Match: {agentic.get('exact_match', 0):.4f}")
    print(f"    - Latency: {agentic.get('latency_ms', 0):.1f} ms")
    print(f"    - Cost: ${agentic.get('total_cost_usd', 0):.4f}")
    print()
    
    paired = summary.get("paired_outcomes", {})
    acc = paired.get("accuracy", {})
    print(f"  Paired Accuracy Outcomes:")
    print(f"    - Agentic Win: {acc.get('agentic_win', 0)}")
    print(f"    - RAG Win: {acc.get('rag_win', 0)}")
    print(f"    - Ties: {acc.get('tie', 0)}")
else:
    print("  Summary not yet available")
EOF

echo ""
echo "View full report:"
echo "  cat $REPORT_PATH"
