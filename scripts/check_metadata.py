#!/usr/bin/env python3
"""Check what metadata is available"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gri_benchmark.data import load_examples
from gri_benchmark.pipelines.agentic_pipeline import AgenticMultiToolPipeline
from gri_benchmark.evidence import SimpleEvidenceRetriever

def main():
    # Load just 3 quantitative samples
    examples = load_examples("data/benchmark/one-table/gri-qa_quant.csv", split="single_table_quantitative")
    examples = examples[:3]
    
    # Create retriever from corpus
    retriever = SimpleEvidenceRetriever.from_jsonl("data/corpus/benchmark_corpus.jsonl")
    
    pipeline = AgenticMultiToolPipeline(strict_mode=True, retriever=retriever)  # Use strict mode with retriever
    
    for i, example in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"[Q{i+1}] {example.question[:70]}...")
        print('='*80)
        
        prediction = pipeline.answer(example)
        
        print(f"\nAnswer: {prediction.answer}")
        print(f"\nMetadata keys: {list(prediction.metadata.keys())}")
        
        # Print each key and its value
        for key, value in prediction.metadata.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in list(value.items())[:5]:  # Show first 5 items
                    print(f"  {k}: {v}")
                if len(value) > 5:
                    print(f"  ... ({len(value)} total)")
            elif isinstance(value, (list, str, bool, int, float)):
                val_str = str(value)[:100]
                print(f"{key}: {val_str}")
            else:
                print(f"{key}: {type(value).__name__}")

if __name__ == "__main__":
    main()
