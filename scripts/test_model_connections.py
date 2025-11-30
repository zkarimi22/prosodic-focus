#!/usr/bin/env python3
"""Quick test script to verify API connections for each model."""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from evaluate script
sys.path.insert(0, str(Path(__file__).parent))

# Import model configuration
from evaluate_conversational_reasoning import MODELS, query_llm, get_client

def test_model(model_id: str):
    """Test a single model with a simple prompt."""
    print(f"\n{'='*60}")
    print(f"Testing {model_id}...")
    print(f"{'='*60}")
    
    try:
        # Check if model is available
        if model_id not in MODELS:
            print(f"❌ Unknown model: {model_id}")
            return False
        
        model_config = MODELS[model_id]
        if not model_config["available"]:
            print(f"❌ Model not available. Provider: {model_config['provider']}")
            print(f"   API key set: {model_config['api_key'] is not None}")
            return False
        
        # Test with a simple prompt
        test_prompt = "Say 'Hello, connection works!' in one sentence."
        print(f"Prompt: {test_prompt}")
        
        response = query_llm(test_prompt, model_id=model_id)
        print(f"✅ Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Test all models."""
    print("Testing API connections for all models...")
    print(f"\nAvailable models: {list(MODELS.keys())}")
    
    results = {}
    for model_id in MODELS.keys():
        results[model_id] = test_model(model_id)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working = [m for m, r in results.items() if r]
    not_working = [m for m, r in results.items() if not r]
    
    if working:
        print(f"\n✅ Working models ({len(working)}):")
        for model in working:
            print(f"   - {model}")
    
    if not_working:
        print(f"\n❌ Not working models ({len(not_working)}):")
        for model in not_working:
            config = MODELS[model]
            print(f"   - {model} (Provider: {config['provider']}, API key: {'set' if config['api_key'] else 'not set'})")
    
    if working:
        print(f"\n💡 You can test with: python3 scripts/evaluate_conversational_reasoning.py --max-pairs 1 --scenario therapy --models {' '.join(working)}")
    else:
        print(f"\n⚠️  No models are working. Please set API keys:")
        print(f"   export OPENAI_API_KEY='your-key'")
        print(f"   export ANTHROPIC_API_KEY='your-key'")
        print(f"   export GEMINI_API_KEY='your-key'")
        print(f"   export DEEPSEEK_API_KEY='your-key'")

if __name__ == "__main__":
    main()

