#!/usr/bin/env python3
"""
Generate a structured biological narrative from a context text file using OpenAI API.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import model configuration
try:
    from model_config import get_model_config
    MODEL_CONFIG_AVAILABLE = True
except ImportError:
    MODEL_CONFIG_AVAILABLE = False
    print("[warn] model_config module not found; using default OpenAI configuration")

def gpt_call(messages: List[Dict[str, str]], *,
             model: str = "gpt-4.1",
             max_tokens: int = 2048,
             temperature: float = 0.2,
             retries: int = 3,
             backoff: float = 5.0) -> str:
    """Wrapper around openai.ChatCompletion.create with simple retry logic."""
    import openai
    global MODEL_CONFIG_AVAILABLE
    
    # Configure OpenAI client based on model_config if available
    if MODEL_CONFIG_AVAILABLE:
        try:
            # Try to get model configuration by name or shortname
            model_config = None
            try:
                # First try by model name
                model_config = get_model_config(model)
            except ValueError:
                # Then try by shortname
                try:
                    model_config = get_model_config(model, by_shortname=True)
                except ValueError:
                    print(f"[warn] Model '{model}' not found in configuration, using default OpenAI settings")
            
            # Apply configuration if found
            if model_config:
                # Save the original API settings to restore later if needed
                original_api_key = openai.api_key
                original_api_base = getattr(openai, "api_base", None)
                
                # Apply the model-specific settings
                openai.api_key = model_config["openai_api_key"]
                openai.api_base = model_config["openai_api_base"]
                model_name = model_config["openai_model"]
                
                print(f"[info] Using model configuration for {model}: {model_name} at {openai.api_base}")
                
                # Update the model name to use
                model = model_name
        except Exception as e:
            print(f"[warn] Error configuring model from YAML: {e}. Using default configuration.")
    
    # Make the API call with retries
    for attempt in range(1, retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message["content"].strip()
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"[warn] OpenAI call failed on attempt {attempt}: {exc}. Retrying in {backoff}s...", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("gpt_call exhausted retries without success")

def generate_narrative(context: str,
                       model: str = "gpt-4",
                       max_tokens: int = 2048,
                       temperature: float = 0.2,
                       retries: int = 3,
                       backoff: float = 5.0) -> str:
    """Generate a structured narrative from context using the LLM."""
    sys_prompt = (
        "You are a knowledgeable biology assistant. "
        "Given unstructured biological context text, generate a structured narrative "
        "with detailed explanations."
    )
    user_prompt = (
        "Expand the following context into a structured narrative that explains the biology:\n\n"
        "1. Entities: Describe the biological entities mentioned.\n"
        "2. Interactions: Describe how these entities interact.\n"
        "3. Biological Effects: Explain the biological consequences of these interactions.\n"
        "4. References: Provide relevant references or citations for the entities and interactions described.\n"
        "5. Biological Reasoning: Provide a reasoning trace that outlines the thought process leading to these conclusions.\n\n"
        "Context:\n"
        f"{context}"
    )
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return gpt_call(
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        backoff=backoff,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a structured biological narrative from a context file using OpenAI API"
    )
    parser.add_argument(
        "-i", "--input-file", required=True,
        help="Path to text file containing unstructured biological context"
    )
    parser.add_argument(
        "-o", "--output-file",
        help="Path to write the structured narrative (defaults to stdout)"
    )
    parser.add_argument(
        "--model", default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Maximum tokens for the OpenAI response (default: 2048)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature for the OpenAI response (default: 0.2)"
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Number of retries for API calls (default: 3)"
    )
    parser.add_argument(
        "--backoff", type=float, default=5.0,
        help="Backoff time in seconds between retries (default: 5.0)"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("[error] OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        context = Path(args.input_file).read_text()
    except Exception as exc:
        print(f"[error] Could not read input file: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[info] Generating structured narrative...")
    narrative = generate_narrative(
        context,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        backoff=args.backoff,
    )

    if args.output_file:
        try:
            Path(args.output_file).write_text(narrative)
            print(f"[info] Narrative written to {args.output_file}")
        except Exception as exc:
            print(f"[error] Could not write output file: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(narrative)

if __name__ == "__main__":
    main()