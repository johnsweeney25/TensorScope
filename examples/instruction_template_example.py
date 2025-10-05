#!/usr/bin/env python3
"""
Example usage of analyze_instruction_sensitivity function
Demonstrates safe usage patterns and best practices
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from BombshellMetrics import BombshellMetrics
import torch


def main():
    """
    Demonstrate instruction template sensitivity analysis
    """
    print("=" * 60)
    print("Instruction Template Sensitivity Analysis - Usage Example")
    print("=" * 60)
    
    # Initialize the metrics analyzer
    metrics = BombshellMetrics()
    
    # Load a small model for demonstration
    # In practice, use your fine-tuned model
    model_name = "gpt2"  # Using small model for example
    print(f"\nLoading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # ========== EXAMPLE 1: Basic Usage ==========
    print("\n" + "=" * 40)
    print("EXAMPLE 1: Basic Instruction Templates")
    print("=" * 40)
    
    # Define the base query/prompt
    base_prompt = "What is the capital of France"
    
    # Define instruction template variations
    # SECURITY NOTE: Only use {query} placeholder
    instruction_templates = [
        "{query}",  # Direct query
        "Please answer: {query}",  # Polite form
        "Question: {query}\nAnswer:",  # Q&A format
        "Help me with this: {query}",  # Help request
        "You are an assistant. {query}",  # Role instruction
        "### Instruction: {query}\n### Response:",  # Alpaca-style
    ]
    
    print(f"\nBase prompt: '{base_prompt}'")
    print(f"Testing {len(instruction_templates)} template variations")
    
    # Run analysis
    try:
        results = metrics.analyze_instruction_sensitivity(
            model=model,
            base_prompt=base_prompt,
            instruction_templates=instruction_templates,
            tokenizer=tokenizer,
            reference_continuation=None  # Let model generate reference
        )
        
        # Display results
        print("\n--- Results ---")
        print(f"Mean KL Divergence: {results['mean_kl_divergence']:.4f}")
        print(f"Max KL Divergence: {results['max_kl_divergence']:.4f}")
        print(f"Sensitivity Score (std): {results['sensitivity_score']:.4f}")
        print(f"Is Fragile: {results['is_fragile']}")
        
        print("\nPer-template results:")
        for i, template_result in enumerate(results['template_results']):
            print(f"  Template {i+1}: KL={template_result['kl_divergence']:.4f}, "
                  f"Perplexity={template_result['perplexity']:.2f}")
    
    except Exception as e:
        print(f"Error in Example 1: {e}")
    
    # ========== EXAMPLE 2: With Fixed Continuation ==========
    print("\n" + "=" * 40)
    print("EXAMPLE 2: Fixed Reference Continuation")
    print("=" * 40)
    
    base_prompt = "Calculate 2 + 2"
    reference_continuation = "equals 4"
    
    instruction_templates = [
        "{query}",
        "Compute: {query}",
        "What is {query}?",
        "Please calculate {query} for me",
    ]
    
    print(f"\nBase prompt: '{base_prompt}'")
    print(f"Reference continuation: '{reference_continuation}'")
    print(f"Testing {len(instruction_templates)} variations")
    
    try:
        results = metrics.analyze_instruction_sensitivity(
            model=model,
            base_prompt=base_prompt,
            instruction_templates=instruction_templates,
            tokenizer=tokenizer,
            reference_continuation=reference_continuation
        )
        
        print("\n--- Results with fixed continuation ---")
        print(f"Sensitivity Score: {results['sensitivity_score']:.4f}")
        
        if results['is_fragile']:
            print("⚠️  Model shows HIGH sensitivity to instruction format!")
            print("    Consider additional instruction-following training")
        else:
            print("✓  Model shows LOW sensitivity to instruction format")
            print("    Instruction-following appears robust")
    
    except Exception as e:
        print(f"Error in Example 2: {e}")
    
    # ========== EXAMPLE 3: Security Best Practices ==========
    print("\n" + "=" * 40)
    print("EXAMPLE 3: Security & Error Handling")
    print("=" * 40)
    
    # These will be rejected by validation
    unsafe_templates = [
        "{query} and also {other}",  # Multiple placeholders - REJECTED
        "Do {task} instead",  # Wrong placeholder name - REJECTED  
        "Calculate {query",  # Unmatched braces - REJECTED
    ]
    
    print("\nDemonstrating input validation:")
    for template in unsafe_templates:
        try:
            results = metrics.analyze_instruction_sensitivity(
                model=model,
                base_prompt="Test",
                instruction_templates=[template],
                tokenizer=tokenizer
            )
        except ValueError as e:
            print(f"✓ Rejected unsafe template: '{template[:30]}...'")
            print(f"  Reason: {e}")
        except TypeError as e:
            print(f"✓ Rejected invalid type")
            print(f"  Reason: {e}")
    
    # ========== EXAMPLE 4: Interpreting Results ==========
    print("\n" + "=" * 40)
    print("INTERPRETING RESULTS")
    print("=" * 40)
    
    print("""
    Sensitivity Score Interpretation:
    - < 0.1:  Very low sensitivity (robust)
    - 0.1-0.3: Low sensitivity (good)
    - 0.3-0.5: Moderate sensitivity (acceptable)
    - 0.5-0.7: High sensitivity (concerning)
    - > 0.7:  Very high sensitivity (fragile)
    
    KL Divergence Interpretation:
    - Near 0: Templates produce similar outputs
    - > 0.5:  Significant difference in outputs
    - > 1.0:  Major difference, potential issues
    
    Use Cases:
    1. Pre-deployment testing: Ensure model handles various prompts
    2. Fine-tuning evaluation: Check if instruction-following improved
    3. Robustness testing: Identify fragile instruction circuits
    4. Template selection: Choose templates with lowest KL divergence
    """)
    
    # ========== EXAMPLE 5: Batch Analysis ==========
    print("\n" + "=" * 40)
    print("EXAMPLE 5: Analyzing Multiple Prompts")
    print("=" * 40)
    
    test_prompts = [
        "What is machine learning",
        "Explain quantum computing",
        "How does photosynthesis work",
    ]
    
    templates = [
        "{query}",
        "Explain: {query}",
        "Tell me about: {query}",
    ]
    
    print(f"\nTesting {len(test_prompts)} different prompts")
    
    all_sensitivities = []
    for prompt in test_prompts:
        try:
            results = metrics.analyze_instruction_sensitivity(
                model=model,
                base_prompt=prompt,
                instruction_templates=templates,
                tokenizer=tokenizer
            )
            all_sensitivities.append(results['sensitivity_score'])
            print(f"  '{prompt[:30]}...': sensitivity = {results['sensitivity_score']:.3f}")
        except Exception as e:
            print(f"  Error with '{prompt[:30]}...': {e}")
    
    if all_sensitivities:
        avg_sensitivity = sum(all_sensitivities) / len(all_sensitivities)
        print(f"\nAverage sensitivity across prompts: {avg_sensitivity:.3f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()