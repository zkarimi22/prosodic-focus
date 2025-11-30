#!/usr/bin/env python3
"""
Conversational Reasoning Evaluation Script

Evaluates whether adding prosodic focus labels improves LLM ability to:
1. Generate better follow-up questions that probe what matters
2. Understand implicit concerns and emotional subtext
3. Follow conversational threads and recognize what's important
4. Make inferences about unstated issues

This is a more sophisticated evaluation than direct Q&A - it tests whether
focus helps the AI be a better conversational partner.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from collections import defaultdict
import statistics

# Try to import optional dependencies
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# DeepSeek uses OpenAI-compatible API via openai package
DEEPSEEK_AVAILABLE = True

# ---------------------------
# CONFIGURATION
# ---------------------------

# API Keys (optional - only needed if using that provider)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Model Configuration (availability checked at runtime)
def get_models_config():
    """Get model configuration with current API key availability."""
    return {
        "gpt-4o-mini": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "available": os.getenv("OPENAI_API_KEY") is not None
        },
        "claude-sonnet-4": {
            "provider": "anthropic",
            "model_name": "claude-sonnet-4-20250514",  # Update with latest version
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "available": ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY") is not None
        },
        "deepseek-chat": {
            "provider": "deepseek",
            "model_name": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "available": DEEPSEEK_AVAILABLE and os.getenv("DEEPSEEK_API_KEY") is not None
        }
    }

# For backward compatibility, create MODELS dict
MODELS = get_models_config()

# Default model list
DEFAULT_MODELS = ["gpt-4o-mini", "claude-sonnet-4", "deepseek-chat"]

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "minimal_pairs"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
TEMPERATURE = 0.7  # Some creativity for question generation
RATE_LIMIT_SLEEP = 1  # Seconds between API calls

# Initialize clients (lazy initialization)
_clients = {}

# ---------------------------
# Scenario Templates
# ---------------------------

SCENARIO_TEMPLATES = {
    "therapy": {
        "context": """You are a therapist conducting a session. The client seems upset or troubled about something, but hasn't explicitly stated what's bothering them. They've just made a statement.""",
        "instruction": """Given the client's statement, generate a thoughtful follow-up question that helps them open up about what's really concerning them. Focus on what seems most important or emotionally significant in their statement.""",
        "evaluation_criteria": "The question should probe the element that was emphasized (focused) in the original statement, as this likely indicates what the client is really concerned about."
    },
    "conflict_resolution": {
        "context": """You are a mediator helping resolve a conflict. One party has made a statement that hints at their underlying concern, but they haven't stated it directly.""",
        "instruction": """Generate a follow-up question that helps uncover the real issue. Pay attention to what seems most important in their statement.""",
        "evaluation_criteria": "The question should explore the focused element, as emphasis often reveals what someone is really upset about."
    },
    "interview": {
        "context": """You are conducting an investigative interview. The person you're interviewing has made a statement that may contain important information, but they're being somewhat evasive.""",
        "instruction": """Generate a follow-up question that probes the most significant or suspicious element in their statement.""",
        "evaluation_criteria": "The question should focus on the emphasized element, as stress often indicates what the speaker considers important or is trying to highlight."
    },
    "story_continuation": {
        "context": """You are helping develop a story. A character has made a statement that hints at something important, but it's not explicitly stated.""",
        "instruction": """Generate a follow-up question or statement that explores what seems most significant or interesting in what was said.""",
        "evaluation_criteria": "The follow-up should explore the focused element, as emphasis in dialogue often signals what's narratively important."
    }
}

# ---------------------------
# Prompt Templates
# ---------------------------

TEXT_ONLY_PROMPT = """{context}

Client/Person: "{sentence}"

{instruction}

Generate your follow-up question:"""

TEXT_WITH_FOCUS_PROMPT = """{context}

Client/Person: "{sentence}"
Focus: {focus_token}

The word "{focus_token}" had prosodic focus (emphasis) in the original speech, indicating it is the most important element.

{instruction}

{evaluation_criteria}

Generate your follow-up question:"""

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
# Question Generation
# ---------------------------

def create_scenario_for_pair(labels: Dict, scenario_type: str = "therapy") -> Dict:
    """
    Create a conversational scenario for a minimal pair.
    
    Returns a dict with scenarios for both subject and object focus conditions.
    """
    sentence = labels["sentence"]
    tokens = labels["tokens"]
    audio_files = labels["audio_files"]
    
    subject_focus_info = audio_files.get("audio_focus_subject.wav", {})
    object_focus_info = audio_files.get("audio_focus_object.wav", {})
    
    if not subject_focus_info or not object_focus_info:
        return None
    
    subject = subject_focus_info["focus_token"]
    obj = object_focus_info["focus_token"]
    
    template = SCENARIO_TEMPLATES[scenario_type]
    
    scenarios = []
    
    # Scenario 1: Subject focus
    scenarios.append({
        "audio_file": "audio_focus_subject.wav",
        "focus_type": "subject",
        "focus_token": subject,
        "sentence": sentence,
        "scenario_type": scenario_type,
        "context": template["context"],
        "instruction": template["instruction"],
        "evaluation_criteria": template["evaluation_criteria"],
        "expected_focus": subject
    })
    
    # Scenario 2: Object focus
    scenarios.append({
        "audio_file": "audio_focus_object.wav",
        "focus_type": "object",
        "focus_token": obj,
        "sentence": sentence,
        "scenario_type": scenario_type,
        "context": template["context"],
        "instruction": template["instruction"],
        "evaluation_criteria": template["evaluation_criteria"],
        "expected_focus": obj
    })
    
    return scenarios

# ---------------------------
# LLM Query Functions
# ---------------------------

def get_client(model_id: str):
    """Get or create API client for a model."""
    if model_id not in MODELS:
        raise ValueError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")
    
    model_config = MODELS[model_id]
    
    if not model_config["available"]:
        raise ValueError(
            f"Model {model_id} is not available. "
            f"Provider: {model_config['provider']}, "
            f"API key set: {model_config['api_key'] is not None}"
        )
    
    provider = model_config["provider"]
    
    # Return cached client if available
    if provider in _clients:
        return _clients[provider]
    
    # Initialize client based on provider
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=OPENAI_API_KEY)
        _clients[provider] = client
        return client
    
    elif provider == "anthropic":
        if not ANTHROPIC_AVAILABLE:
            raise ValueError("anthropic package not installed. Run: pip install anthropic")
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        _clients[provider] = client
        return client
    
    elif provider == "deepseek":
        if not DEEPSEEK_AVAILABLE:
            raise ValueError("openai package required for DeepSeek")
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not set")
        # DeepSeek uses OpenAI-compatible API
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        _clients[provider] = client
        return client
    
    else:
        raise ValueError(f"Unknown provider: {provider}")

def query_llm(prompt: str, model_id: str, max_retries: int = 3) -> str:
    """Query the LLM with retry logic. Supports multiple providers."""
    model_config = MODELS[model_id]
    provider = model_config["provider"]
    model_name = model_config["model_name"]
    
    for attempt in range(max_retries):
        try:
            if provider == "openai" or provider == "deepseek":
                client = get_client(model_id)
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=200
                )
                answer = completion.choices[0].message.content.strip()
                return answer
            
            elif provider == "anthropic":
                client = get_client(model_id)
                message = client.messages.create(
                    model=model_name,
                    max_tokens=200,
                    temperature=TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = message.content[0].text.strip()
                return answer
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(RATE_LIMIT_SLEEP * (attempt + 1))
            else:
                raise
    
    raise Exception(f"Failed after {max_retries} retries")

def generate_followup_question(scenario: Dict, condition: str, model_id: str) -> str:
    """
    Generate a follow-up question under a given condition.
    
    Args:
        scenario: Scenario dict with context, sentence, etc.
        condition: "text_only" or "text_with_focus"
        model_id: Model ID to use for generation
    
    Returns:
        Generated follow-up question
    """
    if condition == "text_only":
        prompt = TEXT_ONLY_PROMPT.format(
            context=scenario["context"],
            sentence=scenario["sentence"],
            instruction=scenario["instruction"]
        )
    elif condition == "text_with_focus":
        prompt = TEXT_WITH_FOCUS_PROMPT.format(
            context=scenario["context"],
            sentence=scenario["sentence"],
            focus_token=scenario["focus_token"],
            instruction=scenario["instruction"],
            evaluation_criteria=scenario["evaluation_criteria"]
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    
    question = query_llm(prompt, model_id=model_id)
    return question

def evaluate_question_relevance(
    sentence: str,
    focus_token: str,
    generated_question: str,
    model_id: str
) -> Dict:
    """
    Evaluate whether the generated question appropriately probes the focused element.
    Uses LLM-as-judge.
    """
    prompt = EVALUATION_PROMPT.format(
        sentence=sentence,
        focus_token=focus_token.upper(),
        question=generated_question
    )
    
    evaluation = query_llm(prompt, model_id=model_id)
    
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
    scenario_type: str = "therapy",
    models: List[str] = None
) -> Dict:
    """
    Run the conversational reasoning evaluation across multiple models.
    
    Args:
        max_pairs: Maximum number of pairs to evaluate (None = all)
        scenario_type: Type of scenario to use (therapy, conflict_resolution, etc.)
        models: List of model IDs to test (default: all available models)
    
    Returns:
        Results dictionary with metrics and per-pair results
    """
    # Refresh model availability at runtime
    MODELS = get_models_config()
    
    if models is None:
        # Use all available models
        models = [model_id for model_id, config in MODELS.items() if config["available"]]
        if not models:
            raise ValueError("No models available. Please set API keys for at least one model.")
    
    # Validate models
    for model_id in models:
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}. Available: {list(MODELS.keys())}")
        if not MODELS[model_id]["available"]:
            raise ValueError(f"Model {model_id} is not available. Check API key.")
    
    print("Loading minimal pairs...")
    all_pairs = load_all_pairs()
    
    if max_pairs:
        all_pairs = all_pairs[:max_pairs]
    
    print(f"Found {len(all_pairs)} pairs to evaluate")
    print(f"Testing models: {', '.join(models)}")
    print(f"Scenario type: {scenario_type}")
    print(f"Conditions: text_only, text_with_focus")
    print("=" * 60)
    
    results = {
        "models": models,
        "scenario_type": scenario_type,
        "total_pairs": len(all_pairs),
        "conditions": ["text_only", "text_with_focus"],
        "per_pair_results": [],
        "summary": {}
    }
    
    # Stats per model and condition
    stats = {}
    for model_id in models:
        stats[model_id] = {
            "text_only": {"relevant": 0, "total": 0, "questions": []},
            "text_with_focus": {"relevant": 0, "total": 0, "questions": []}
        }
    
    for pair_idx, (pair_dir, labels) in enumerate(all_pairs, 1):
        print(f"\n[{pair_idx}/{len(all_pairs)}] {pair_dir.name}")
        print(f"  Sentence: {labels['sentence']}")
        
        # Create scenarios for this pair
        scenarios = create_scenario_for_pair(labels, scenario_type)
        
        if not scenarios:
            print("  ⚠️  Could not create scenarios, skipping")
            continue
        
        pair_result = {
            "pair_id": pair_dir.name,
            "sentence": labels["sentence"],
            "scenarios": []
        }
        
        for scenario in scenarios:
            print(f"\n  Scenario: {scenario['focus_type']} focus on '{scenario['focus_token']}'")
            
            scenario_result = {
                "focus_type": scenario["focus_type"],
                "focus_token": scenario["focus_token"],
                "models": {}
            }
            
            for model_id in models:
                print(f"    Model: {model_id}")
                model_result = {}
                
                # Generate questions for both conditions
                for condition in ["text_only", "text_with_focus"]:
                    print(f"      Generating {condition} question...")
                    try:
                        question = generate_followup_question(scenario, condition, model_id=model_id)
                        print(f"      Question: {question[:80]}...")
                    except Exception as e:
                        print(f"      ❌ Error generating question for {model_id} ({condition}): {e}")
                        question = None
                    
                    # Evaluate relevance (use first model for evaluation to save costs)
                    eval_model_id = models[0]  # Use first model for evaluation
                    try:
                        print(f"      Evaluating relevance (using {eval_model_id})...")
                        evaluation = evaluate_question_relevance(
                            sentence=scenario["sentence"],
                            focus_token=scenario["focus_token"],
                            generated_question=question or "",
                            model_id=eval_model_id
                        )
                    except Exception as e:
                        print(f"      ❌ Error evaluating relevance for {model_id} ({condition}): {e}")
                        evaluation = {
                            "evaluation_text": f"ERROR: {e}",
                            "is_relevant": False,
                            "focus_token": scenario["focus_token"]
                        }
                    
                    # Always record stats, even on errors (as incorrect)
                    stats[model_id][condition]["total"] += 1
                    if evaluation["is_relevant"]:
                        stats[model_id][condition]["relevant"] += 1
                    
                    stats[model_id][condition]["questions"].append({
                        "pair_id": pair_dir.name,
                        "focus_token": scenario["focus_token"],
                        "question": question,
                        "is_relevant": evaluation["is_relevant"],
                        "evaluation": evaluation["evaluation_text"]
                    })
                    
                    model_result[condition] = {
                        "question": question,
                        "evaluation": evaluation
                    }
                    
                    status = "✓" if evaluation["is_relevant"] else "✗"
                    print(f"      Relevance: {status}")
                    
                    time.sleep(RATE_LIMIT_SLEEP)
                
                scenario_result["models"][model_id] = model_result
            
            pair_result["scenarios"].append(scenario_result)
        
        results["per_pair_results"].append(pair_result)
        
        # Print running stats
        print(f"\n  Running stats:")
        for model_id in models:
            text_only_pct = stats[model_id]["text_only"]["relevant"] / stats[model_id]["text_only"]["total"] if stats[model_id]["text_only"]["total"] > 0 else 0
            text_focus_pct = stats[model_id]["text_with_focus"]["relevant"] / stats[model_id]["text_with_focus"]["total"] if stats[model_id]["text_with_focus"]["total"] > 0 else 0
            print(f"    {model_id}: Text-only {text_only_pct:.2%}, Text+Focus {text_focus_pct:.2%}")
    
    # Calculate final summary statistics per model
    for model_id in models:
        results["summary"][model_id] = {}
        for condition in ["text_only", "text_with_focus"]:
            model_stats = stats[model_id][condition]
            if model_stats["total"] > 0:
                relevance_rate = model_stats["relevant"] / model_stats["total"]
                results["summary"][model_id][condition] = {
                    "relevance_rate": relevance_rate,
                    "relevant": model_stats["relevant"],
                    "total": model_stats["total"]
                }
        
        # Calculate improvement for this model
        if "text_only" in results["summary"][model_id] and "text_with_focus" in results["summary"][model_id]:
            baseline_rate = results["summary"][model_id]["text_only"]["relevance_rate"]
            focus_rate = results["summary"][model_id]["text_with_focus"]["relevance_rate"]
            improvement = focus_rate - baseline_rate
            improvement_pct = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
            results["summary"][model_id]["improvement"] = {
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
    
    for model_id in results["models"]:
        print(f"\n{model_id}:")
        summary = results["summary"][model_id]
        
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
        description="Evaluate conversational reasoning with and without focus labels"
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
        default=None,
        help=f"Models to test (default: all available). Options: {', '.join(MODELS.keys())}"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="therapy",
        choices=list(SCENARIO_TEMPLATES.keys()) + ["all"],
        help=f"Scenario type (default: therapy, or 'all' for all scenarios)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: auto-generated timestamp)"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    all_results = []
    if args.scenario == "all":
        scenarios = list(SCENARIO_TEMPLATES.keys())
    else:
        scenarios = [args.scenario]
    
    for scen in scenarios:
        print(f"\nRunning scenario: {scen}")
        results = run_evaluation(max_pairs=args.max_pairs, scenario_type=scen, models=args.models)
        all_results.append(results)
        
        # Save results
        if args.output and len(scenarios) == 1:
            output_file = Path(args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = OUTPUT_DIR / f"conversational_eval_{scen}_{timestamp}.json"
        
        save_results(results, output_file)
        
        # Print summary for this scenario
        print_summary(results)
    
    return all_results if len(all_results) > 1 else all_results[0]

if __name__ == "__main__":
    main()

