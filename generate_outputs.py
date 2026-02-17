"""
Generate model outputs for AlpacaEval using vLLM.

Usage:
    python generate_outputs.py --model /path/to/model --output outputs.json
    python generate_outputs.py --model Qwen/Qwen2.5-7B-Instruct --output outputs.json
    python generate_outputs.py --model /path/to/model --output outputs.json --gpus 2 --batch_size 512
"""

import argparse
import json
import os

import datasets
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_instructions():
    dataset = datasets.load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval_gpt4_baseline",
        trust_remote_code=True,
    )["eval"]
    return list(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="模型路径或 HuggingFace repo")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径")
    parser.add_argument("--gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256, help="vLLM max_num_seqs")
    args = parser.parse_args()

    print(f"Loading dataset...")
    examples = load_instructions()
    instructions = [ex["instruction"] for ex in examples]
    print(f"Loaded {len(instructions)} instructions.")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.gpus,
        max_num_seqs=args.batch_size,
        dtype="bfloat16",
    )

    # Apply chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for instr in instructions
    ]

    print(f"Running inference on {len(prompts)} prompts...")
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    outputs = llm.generate(prompts, sampling_params)
    completions = [out.outputs[0].text for out in outputs]

    model_name = os.path.basename(args.model.rstrip("/"))
    results = []
    for ex, output in zip(examples, completions):
        results.append({
            "instruction": ex["instruction"],
            "output": output,
            "generator": model_name,
            "dataset": ex.get("dataset", ""),
            "datasplit": ex.get("datasplit", "eval"),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} outputs to {args.output}")


if __name__ == "__main__":
    main()
