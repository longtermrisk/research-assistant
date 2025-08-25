"""
Milestone 3: Evaluation Framework for FileSystemStore

This script provides:
1. Batch ingestion of all files in a directory (parallel processing)
2. Gitignore support to exclude ignored files
3. Evaluation mode with JSON test cases and ranking results
4. Interactive query mode showing document sources
5. Comprehensive performance metrics
"""

import asyncio
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import time
from rag import FileSystemStore, OpenAIEmbedder, NoAdditionalKeys, Document, OpenAiAnnotator, StandardReranker
from dtypes import TextBlock


async def run_evaluation(store: FileSystemStore, eval_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run evaluation using test cases from JSON file."""
    
    # Load evaluation data
    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    print(f"Running evaluation with {len(eval_data)} test cases...")
    
    results = []
    total_correct = 0
    total_files = 0
    
    for test_case in eval_data:
        test_id = test_case['test_case_id']
        query = test_case['query']
        correct_sources = test_case['correct']
        
        print(f"\nTest Case {test_id}: {query[:60]}...")
        
        # Query the store with a large top_k to ensure we get all relevant results
        query_content = [TextBlock(text=query)]
        retrieved_docs = await store.query(query_content, top_k=1000)
        
        # Create mapping of source to rank
        source_to_rank = {}
        for rank, doc in enumerate(retrieved_docs, 1):
            source_to_rank[doc.meta.source] = rank
        
        # Process correct sources
        correct_results = []
        ranks = []
        
        for correct_source in correct_sources:
            if correct_source in source_to_rank:
                rank = source_to_rank[correct_source]
                correct_results.append({
                    "source": correct_source,
                    "rank": rank,
                    "found": True
                })
                ranks.append(rank)
                total_correct += 1
                print(f"  ✅ {correct_source} found at rank {rank}")
            else:
                correct_results.append({
                    "source": correct_source,
                    "rank": None,
                    "found": False
                })
                print(f"  ❌ {correct_source} not found in top 50 results")
                breakpoint()
        
        total_files += len(correct_sources)
        
        # Calculate min_top_k (maximum rank needed to get all correct files)
        min_top_k = max(ranks) if ranks else None
        
        # Calculate metrics for this test case
        precision_at_5 = len([r for r in ranks if r <= 5]) / min(5, len(retrieved_docs)) if retrieved_docs else 0
        recall_at_5 = len([r for r in ranks if r <= 5]) / len(correct_sources) if correct_sources else 0
        
        result = {
            "test_case_id": test_id,
            "query": query,
            "correct": correct_results,
            "min_top_k": min_top_k,
            "total_retrieved": len(retrieved_docs),
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5
        }
        
        results.append(result)
    
    # Calculate overall metrics
    overall_recall = total_correct / total_files if total_files > 0 else 0
    avg_min_top_k = sum(r['min_top_k'] for r in results if r['min_top_k'] is not None) / len([r for r in results if r['min_top_k'] is not None])
    
    eval_results = {
        "evaluation_summary": {
            "total_test_cases": len(eval_data),
            "total_correct_files": total_files,
            "total_found_files": total_correct,
            "overall_recall": overall_recall,
            "average_min_top_k": avg_min_top_k
        },
        "test_cases": results
    }
    
    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n📄 Results saved to {output_path}")
    
    # Print summary
    print(f"\n📊 EVALUATION SUMMARY")
    print(f"=" * 50)
    print(f"Test cases: {len(eval_data)}")
    print(f"Total files to find: {total_files}")
    print(f"Files found: {total_correct}")
    print(f"Overall recall: {overall_recall:.2%}")
    print(f"Average min_top_k: {avg_min_top_k:.1f}")
    
    return eval_results


async def interactive_mode(store: FileSystemStore):
    """Run interactive query mode."""
    
    print("\n" + "="*60)
    print("🔍 INTERACTIVE QUERY MODE")
    print("="*60)
    print("Ask questions about the ingested documents!")
    print("Commands:")
    print("  'stats' - Show knowledge base statistics")
    print("  'quit' - Exit")
    print()
    
    while True:
        try:
            user_input = input("❓ Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'stats':
                stats = store.get_stats()
                print(f"\n📊 Knowledge Base Statistics:")
                print(f"   Documents: {stats['total_documents']}")
                print(f"   Keys: {stats['total_keys']}")
                print(f"   Embedder: {stats['embedder_id']}")
                print(f"   Directory: {stats['root_directory']}")
                continue
            
            if not user_input:
                continue
            
            print("\n🔎 Searching...")
            query_content = [TextBlock(text=user_input)]
            results = await store.query(query_content, top_k=4)
            
            if not results:
                print("❌ No relevant documents found.")
                continue
            
            print(f"\n📋 Found {len(results)} relevant document(s):")
            print("-" * 60)
            
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. 📄 {doc.meta.title}")
                print(f"   📁 Source: {doc.meta.source}")
                print(f"   📅 Created: {doc.meta.created_at.strftime('%Y-%m-%d %H:%M')}")
                
                # Show content preview
                text_content = ""
                for block in doc.content:
                    if isinstance(block, TextBlock):
                        text_content = block.text
                        break
                
                # Show first few lines
                lines = text_content.strip().split('\n')[:3]
                preview = '\n'.join(lines)
                print(f"   📝 Preview:")
                for line in preview.split('\n'):
                    print(f"      {line.strip()[:80]}{'...' if len(line.strip()) > 80 else ''}")
                if len(text_content.split('\n')) > 3:
                    print("      ...")
            
            print("\n" + "-" * 60)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


async def main():
    """Main function with argument parsing."""
    
    parser = argparse.ArgumentParser(
        description="Light-RAG Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest directory and run interactive mode
  python example_filesystem.py --ingest-dir ./docs --store-dir ./knowledge_base
  
  # Run evaluation
  python example_filesystem.py --store-dir ./knowledge_base --eval-file eval.json --output results.json
  
  # Ingest and evaluate in one go
  python example_filesystem.py --ingest-dir ./docs --store-dir ./kb --eval-file eval.json
  
  # Ingest without respecting .gitignore
  python example_filesystem.py --ingest-dir ./docs --store-dir ./kb --no-gitignore
  
  # Update existing documents instead of ignoring duplicates
  python example_filesystem.py --ingest-dir ./docs --store-dir ./kb --existing-source update
        """
    )
    
    parser.add_argument('--store-dir', required=True,
                       help='Directory for the FileSystemStore')
    parser.add_argument('--ingest-dir', 
                       help='Directory to ingest files from (optional)')
    parser.add_argument('--eval-file',
                       help='JSON file with evaluation test cases (optional)')
    parser.add_argument('--output',
                       help='Output file for evaluation results (optional)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Maximum parallel workers for ingestion (default: 10)')
    parser.add_argument('--no-gitignore', action='store_true',
                       help='Ignore .gitignore patterns and ingest all files')
    parser.add_argument('--existing-source', choices=['ignore', 'update'], default='ignore',
                       help='How to handle documents with existing sources (default: ignore)')
    parser.add_argument('--context',
                       help='Path to context file (optional)')
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: Please set OPENAI_API_KEY environment variable")
        print("Get your API key from: https://platform.openai.com/api-keys")
        return
    
    print("🚀 Light-RAG Evaluation Framework")
    print("=" * 50)
    
    try:
        # Load or create store
        if os.path.exists(args.store_dir) and os.path.exists(os.path.join(args.store_dir, "store_config.json")):
            print(f"📚 Loading existing store from {args.store_dir}")
            store = FileSystemStore.load(args.store_dir)
            stats = store.get_stats()
            print(f"   Loaded {stats['total_documents']} documents")
        else:
            print(f"🆕 Creating new store in {args.store_dir}")
            embedder = OpenAIEmbedder(model="text-embedding-3-small")
            context = None
            if args.context:
                context_path = Path(args.context).resolve()
                if not context_path.exists():
                    raise FileNotFoundError(f"Context file {args.context} does not exist")
                with open(context_path, 'r', encoding='utf-8') as f:
                    context = [TextBlock(text=f.read())]
            annotator = OpenAiAnnotator(model="gpt-4.1-mini", n_keys=10)
            store = FileSystemStore.create(embedder=embedder, annotator=annotator, reranker=StandardReranker(), root_dir=args.store_dir, context=context)
        
        # Ingest directory if specified
        if args.ingest_dir:
            respect_gitignore = not args.no_gitignore
            gitignore_status = "respecting .gitignore" if respect_gitignore else "ignoring .gitignore"
            duplicate_status = f"existing sources: {args.existing_source}"
            
            print(f"\n📁 Ingesting files from {args.ingest_dir}")
            print(f"   Options: {gitignore_status}, {duplicate_status}")
            
            ingested_docs = await store.ingest_dir(
                directory=args.ingest_dir,
                existing_source=args.existing_source,
                max_workers=args.max_workers,
                respect_gitignore=respect_gitignore
            )
            
            # Show final stats
            stats = store.get_stats()
            print(f"\n📊 Final store statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Total keys: {stats['total_keys']}")
        
        # Run evaluation if specified
        if args.eval_file:
            print(f"\n🧪 Running evaluation from {args.eval_file}")
            await run_evaluation(store, args.eval_file, args.output)
        else:
            # Run interactive mode
            await interactive_mode(store)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())