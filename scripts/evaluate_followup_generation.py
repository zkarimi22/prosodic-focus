#!/usr/bin/env python3
"""
Generic Follow-up Question Generation Evaluation

A simple, generic evaluation that tests whether focus labels help LLMs
identify the most important thread to explore in follow-up questions.

Prompt: "You receive the following statement. Determine the best thread 
to ask a follow-up question on and ask it."

Evaluates across multiple LLMs to compare performance.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from collections import defaultdict

# ---------------------------
# CONFIGURATION
# ---------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Please set OPENAI_API_KEY environment variable. "
        "Run: export OPENAI_API_KEY='your-key-here'"
    )

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "minimal_pairs"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]  # Can add more models
TEMPERATURE = 0.7
RATE_LIMIT_SLEEP = 1

# Initialize client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Prompt Templates
# ---------------------------

TEXT_ONLY_PROMPT = """You receive the following statement. Determine the best thread to ask a follow-up question on and ask it.

Statement: "{sentence}"

Generate a thoughtful follow-up question:"""

TEXT_WITH_FOCUS_PROMPT = """You receive the following statement. Determine the best thread to ask a follow-up question on and ask it.

Statement: "{sentence}"
Focus: {focus_token}

The word "{focus_token}" had prosodic focus (emphasis) in the original speech, indicating it is the most important element.

Generate a thoughtful follow-up question:"""

# Improved evaluation criteria - more reasonable
EVALUATION_PROMPT = """Evaluate whether the follow-up question appropriately explores the focused element.

Original statement: "{sentence}"
Focus: {focus_token}
Generated question: "{question}"

Does this question explore, probe, or ask about the focused element ({focus_token}) or things directly related to it?

Consider the question relevant if it:
- Asks about the focused word itself
- Asks about attributes, details, or aspects of the focused word
- Probes why the focused word is significant or important
- Explores the relationship between the focused word and other elements

Respond with only "YES" or "NO" followed by a brief explanation (1-2 sentences)."""

# ---------------------------
# LLM Query Functions
# ---------------------------

def query_llm(prompt: str, model: str, max_retries: int = 3) -> str:
    """Query the LLM with retry logic."""
    for attempt in range(max_retries):
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=200
            )
            answer = completion.choices[0].message.content.strip()
            return answer
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
            else:
                raise
    
    raise Exception(f"Failed after {max_retries} retries")

def generate_followup_question(
    sentence: str,
    focus_token: Optional[str],
    condition: str,
    model: str
) -> str:
    """
    Generate a follow-up question under a given condition.
    
    Args:
        sentence: The original sentence
        focus_token: The word with focus (None for text-only)
        condition: "text_only" or "text_with_focus"
        model: Model to use
    
    Returns:
        Generated follow-up question
    """
    if condition == "text_only":
        prompt = TEXT_ONLY_PROMPT.format(sentence=sentence)
    elif condition == "text_with_focus":
        prompt = TEXT_WITH_FOCUS_PROMPT.format(sentence=sentence, focus_token=focus_token)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    question = query_llm(prompt, model=model)
    return question

def evaluate_question_relevance(
    sentence: str,
    focus_token: str,
    generated_question: str,
    model: str
) -> Dict:
    """
    Evaluate whether the generated question appropriately explores the focused element.
    Uses LLM-as-judge with improved criteria.
    """
    prompt = EVALUATION_PROMPT.format(
        sentence=sentence,
        focus_token=focus_token.upper(),
        question=generated_question
    )
    
    evaluation = query_llm(prompt, model=model)
    
    # Parse evaluation
    is_relevant = evaluation.strip().upper().startswith("YES")
    
    return {
        "evaluation_text": evaluation,
        "is_relevant": is_relevant,
        "focus_token": focus_token
    }

# ---------------------------
# Main Evaluation Pipeline
# ---------------------------

def load_all_pairs() -> List[Tuple[Path, Dict]]:
    """Load all minimal pairs from the data directory."""
    pairs = []
    for pair_dir in sorted(DATA_DIR.glob("pair_*")):
        labels_file = pair_dir / "labels.json"
        if labels_file.exists():
            try:
                with open(labels_file, "r") as f:
                    labels = json.load(f)
                pairs.append((pair_dir, labels))
            except Exception as e:
                print(f"Warning: Could not load {labels_file}: {e}")
    return pairs

def run_evaluation(
    max_pairs: Optional[int] = None,
    models: List[str] = None
) -> Dict:
    """
    Run the generic follow-up question evaluation across multiple models.
    
    Args:
        max_pairs: Maximum number of pairs to evaluate (None = all)
        models: List of models to test (default: MODELS)
    
    Returns:
        Results dictionary with metrics and per-pair results
    """
    if models is None:
        models = MODELS
    
    print("Loading minimal pairs...")
    all_pairs = load_all_pairs()
    
    if max_pairs:
        all_pairs = all_pairs[:max_pairs]
    
    print(f"Found {len(all_pairs)} pairs to evaluate")
    print(f"Testing models: {', '.join(models)}")
    print(f"Conditions: text_only, text_with_focus")
    print("=" * 60)
    
    results = {
        "models": models,
        "total_pairs": len(all_pairs),
        "conditions": ["text_only", "text_with_focus"],
        "per_pair_results": [],
        "summary": {}
    }
    
    # Stats per model and condition
    stats = {}
    for model in models:
        stats[model] = {
            "text_only": {"relevant": 0, "total": 0},
            "text_with_focus": {"relevant": 0, "total": 0}
        }
    
    for pair_idx, (pair_dir, labels) in enumerate(all_pairs, 1):
        print(f"\n[{pair_idx}/{len(all_pairs)}] {pair_dir.name}")
        print(f"  Sentence: {labels['sentence']}")
        
        audio_files = labels["audio_files"]
        subject_focus_info = audio_files.get("audio_focus_subject.wav", {})
        object_focus_info = audio_files.get("audio_focus_object.wav", {})
        
        if not subject_focus_info or not object_focus_info:
            print("  ⚠️  Missing focus info, skipping")
            continue
        
        pair_result = {
            "pair_id": pair_dir.name,
            "sentence": labels["sentence"],
            "scenarios": []
        }
        
        # Test both subject and object focus
        for focus_info, focus_type in [(subject_focus_info, "subject"), (object_focus_info, "object")]:
            focus_token = focus_info["focus_token"]
            print(f"\n  Focus: {focus_type} ('{focus_token}')")
            
            scenario_result = {
                "focus_type": focus_type,
                "focus_token": focus_token,
                "models": {}
            }
            
            for model in models:
                print(f"    Model: {model}")
                model_result = {}
                
                # Generate questions for both conditions
                for condition in ["text_only", "text_with_focus"]:
                    print(f"      Generating {condition} question...")
                    question = generate_followup_question(
                        sentence=labels["sentence"],
                        focus_token=focus_token if condition == "text_with_focus" else None,
                        condition=condition,
                        model=model
                    )
                    print(f"      Question: {question[:80]}...")
                    
                    # Evaluate relevance
                    print(f"      Evaluating relevance...")
                    evaluation = evaluate_question_relevance(
                        sentence=labels["sentence"],
                        focus_token=focus_token,
                        generated_question=question,
                        model=model
                    )
                    
                    stats[model][condition]["total"] += 1
                    if evaluation["is_relevant"]:
                        stats[model][condition]["relevant"] += 1
                    
                    model_result[condition] = {
                        "question": question,
                        "evaluation": evaluation
                    }
                    
                    status = "✓" if evaluation["is_relevant"] else "✗"
                    print(f"      Relevance: {status}")
                    
                    time.sleep(RATE_LIMIT_SLEEP)
                
                scenario_result["models"][model] = model_result
            
            pair_result["scenarios"].append(scenario_result)
        
        results["per_pair_results"].append(pair_result)
        
        # Print running stats
        print(f"\n  Running stats:")
        for model in models:
            text_only_pct = stats[model]["text_only"]["relevant"] / stats[model]["text_only"]["total"] if stats[model]["text_only"]["total"] > 0 else 0
            text_focus_pct = stats[model]["text_with_focus"]["relevant"] / stats[model]["text_with_focus"]["total"] if stats[model]["text_with_focus"]["total"] > 0 else 0
            print(f"    {model}: Text-only {text_only_pct:.2%}, Text+Focus {text_focus_pct:.2%}")
    
    # Calculate final summary statistics
    for model in models:
        results["summary"][model] = {}
        for condition in ["text_only", "text_with_focus"]:
            model_stats = stats[model][condition]
            if model_stats["total"] > 0:
                relevance_rate = model_stats["relevant"] / model_stats["total"]
                results["summary"][model][condition] = {
                    "relevance_rate": relevance_rate,
                    "relevant": model_stats["relevant"],
                    "total": model_stats["total"]
                }
        
        # Calculate improvement for this model
        if "text_only" in results["summary"][model] and "text_with_focus" in results["summary"][model]:
            baseline_rate = results["summary"][model]["text_only"]["relevance_rate"]
            focus_rate = results["summary"][model]["text_with_focus"]["relevance_rate"]
            improvement = focus_rate - baseline_rate
            improvement_pct = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
            results["summary"][model]["improvement"] = {
                "absolute": improvement,
                "relative_percent": improvement_pct
            }
    
    return results

def save_results(results: Dict, output_file: Path):
    """Save evaluation results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

def print_summary(results: Dict):
    """Print a summary of the evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for model in results["models"]:
        print(f"\n{model}:")
        summary = results["summary"][model]
        
        if "text_only" in summary:
            baseline = summary["text_only"]
            print(f"  Baseline (Text-only):")
            print(f"    Relevance Rate: {baseline['relevance_rate']:.2%} ({baseline['relevant']}/{baseline['total']})")
        
        if "text_with_focus" in summary:
            focus = summary["text_with_focus"]
            print(f"  With Focus Labels (Text+Focus):")
            print(f"    Relevance Rate: {focus['relevance_rate']:.2%} ({focus['relevant']}/{focus['total']})")
        
        if "improvement" in summary:
            imp = summary["improvement"]
            print(f"  Improvement:")
            print(f"    Absolute: {imp['absolute']:+.2%}")
            print(f"    Relative: {imp['relative_percent']:+.1f}%")
    
    print("\n" + "=" * 60)

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate generic follow-up question generation with and without focus labels"
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum number of pairs to evaluate (default: all)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODELS,
        help=f"Models to test (default: {MODELS})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated timestamp)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = run_evaluation(max_pairs=args.max_pairs, models=args.models)
    
    # Save results
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"followup_generation_{timestamp}.json"
    
    save_results(results, output_file)
    
    # Print summary
    print_summary(results)
    
    return results

if __name__ == "__main__":
    main()

