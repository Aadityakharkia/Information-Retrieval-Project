"""Interactive Command-Line Interface for Trustworthy Medical RAG.
Provides deep inspectability into intermediate IR outputs, rankings, and citation audits.
"""
import sys
import argparse
from typing import Optional

from medrag.pipeline import TrustworthyMedicalRAGPipeline
from medrag.config import SUPPORTED_GENERATOR_MODELS


def print_banner():
    print("""
================================================================================
   TRUSTWORTHY MEDICAL RAG on MedQuAD (NIH Health Corpus)
   CSD358 IR Hackathon - Track 1: RAG & Trustworthy Answers
================================================================================
    """)


def format_ir_inspection(diag: dict):
    print("\n" + "-"*70)
    print(" [IR INSPECTION: INTERMEDIATE RETRIEVAL STATE]")
    print("-"*70)
    print(f"Retriever: {diag.get('retriever')}")
    if "stems" in diag:
        print(f"Tokenized Stems: {diag.get('stems')}")
    if "query_norm" in diag:
        print(f"Query Euclidean Norm (|q|): {diag.get('query_norm')}")
    if "term_diagnostics" in diag:
        print("\nTerm Postings & Weight Diagnostics (lnc.ltc):")
        print(f" {'Term':<15} {'DF':<8} {'IDF':<8} {'Norm LTC':<10} {'Postings Count':<15}")
        print(" " + "-"*60)
        for term, d in diag["term_diagnostics"].items():
            print(f" {term:<15} {d.get('df', 0):<8} {d.get('idf', 0.0):<8.4f} {d.get('normalized_ltc_weight', 0.0):<10.4f} {d.get('postings_count', 0):<15}")
    if "top_fused_results" in diag:
        print("\nReciprocal Rank Fusion (RRF) Decomposition:")
        print(f" {'Chunk ID':<18} {'Fused Rank':<12} {'Sparse (R, Contrib)':<22} {'Dense (R, Contrib)':<22}")
        print(" " + "-"*75)
        for row in diag["top_fused_results"]:
            s_info = f"#{row['sparse_rank']} ({row['sparse_rrf_contrib']:.5f})" if row['sparse_rank'] else "None"
            d_info = f"#{row['dense_rank']} ({row['dense_rrf_contrib']:.5f})" if row['dense_rank'] else "None"
            print(f" {row['chunk_id']:<18} #{row['fused_rank']:<11} {s_info:<22} {d_info:<22}")
    print("-"*70)


def format_citation_audit(audit: dict):
    print("\n" + "-"*70)
    print(f" [CITATION AUDIT: {audit.get('verdict')}] Faithfulness Score: {audit.get('faithfulness_score', 0.0)*100:.1f}%")
    print(f" Supported: {audit.get('supported_count')} | Partially Supported: {audit.get('partially_supported_count')} | Unsupported: {audit.get('unsupported_count')} | Uncited: {audit.get('uncited_count')}")
    print("-"*70)
    for s in audit.get("sentence_audits", []):
        verdict = s["verdict"]
        badge = (
            "\033[92m[SUPPORTED]\033[0m" if verdict == "SUPPORTED"
            else "\033[93m[PARTIAL]\033[0m" if verdict == "PARTIALLY_SUPPORTED"
            else "\033[91m[UNSUPPORTED]\033[0m" if verdict == "UNSUPPORTED"
            else "\033[94m[DISCLAIMER]\033[0m" if verdict == "DISCLAIMER"
            else "\033[95m[UNCITED]\033[0m"
        )
        print(f"\n Claim {s['sentence_index']}: {s['clean_text']}")
        print(f" Status: {badge} | Citations: {s['citations']} | Score: {s['confidence_score']:.3f} (Cosine: {s['cosine_similarity']:.3f}, Term Cov: {s['term_coverage']:.3f})")
        print(f" Note: {s['explanation']}")
    print("-"*70)


def run_cli():
    print_banner()
    parser = argparse.ArgumentParser(description="Trustworthy Medical RAG CLI")
    parser.add_argument("--retriever", choices=["inverted_index", "bm25", "dense", "hybrid"], default="inverted_index", help="Retrieval algorithm")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=None, help="Refusal score threshold")
    parser.add_argument("--model", choices=SUPPORTED_GENERATOR_MODELS, default="llama-3.3-70b-versatile", help="Generator LLM")
    parser.add_argument("--query", type=str, default=None, help="Single query mode")
    args = parser.parse_args()

    print("Loading pipeline components (corpus, indexes, models)...")
    pipeline = TrustworthyMedicalRAGPipeline()
    pipeline.initialize()

    if args.query:
        print(f"\nProcessing Query: '{args.query}'\n")
        res = pipeline.answer_question(
            question=args.query,
            retriever_type=args.retriever,
            top_k=args.top_k,
            threshold=args.threshold,
            generator_model=args.model
        )
        format_ir_inspection(res["ir_diagnostics"])
        print("\n" + "="*70)
        print(" [GENERATED ANSWER]")
        print("="*70)
        print(res["answer"])
        print("="*70)
        format_citation_audit(res["citation_audit"])
        return

    print("\nPipeline Ready! Type 'exit' or 'quit' to stop.\n")
    current_retriever = args.retriever
    
    while True:
        try:
            user_input = input(f"\n[IR-RAG ({current_retriever})] Enter medical question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
                
            if user_input.startswith(":set_retriever "):
                current_retriever = user_input.split(" ", 1)[1].strip()
                print(f"Switched retriever to: {current_retriever}")
                continue

            res = pipeline.answer_question(
                question=user_input,
                retriever_type=current_retriever,
                top_k=args.top_k,
                threshold=args.threshold,
                generator_model=args.model
            )

            # Display inspectable diagnostics
            format_ir_inspection(res["ir_diagnostics"])

            # Display Answer
            print("\n" + "="*70)
            if res["refused"]:
                print(" [SYSTEM REFUSAL - LOW CONFIDENCE]")
                print(f" Reason: {res['refusal_reason']}")
            else:
                print(f" [GENERATED ANSWER - {res['generator_model']}]")
            print("="*70)
            print(res["answer"])
            print("="*70)

            # Display Citation Audit
            if not res["refused"]:
                format_citation_audit(res["citation_audit"])

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    run_cli()
