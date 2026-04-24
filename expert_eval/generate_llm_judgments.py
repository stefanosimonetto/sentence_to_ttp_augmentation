from pathlib import Path
import json
import os
import time

from openai import OpenAI


JUDGMENTS = ["plausible", "mild", "negative"]
DEFAULT_MODEL = os.environ.get("EXPERT_EVAL_LLM_MODEL", "gpt-4o-mini")

APPROACH_FILES = {
    "pseudo_label": [
        "audit_sample_cti_rev1.json",
        "audit_sample_cti_rev2.json",
    ],
    "embeddings": [
        "audit_sample_embeddings_rev3.json",
        "audit_sample_embeddings_rev2.json",
    ],
    "hierarchy": [
        "audit_sample_hierarchy_rev3.json",
        "audit_sample_hierarchy_rev1.json",
    ],
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_eval_dir() -> Path:
    all_configured_files = [file_name for files in APPROACH_FILES.values() for file_name in files]
    eval_dir_candidates = [Path("."), Path("expert_eval"), Path.cwd(), Path.cwd() / "expert_eval", Path.cwd().parent / "expert_eval"]

    for candidate in eval_dir_candidates:
        if all((candidate / file_name).exists() for file_name in all_configured_files):
            return candidate

    raise FileNotFoundError("Could not find expert_eval annotation files.")


def build_examples(eval_dir: Path):
    examples_by_key = {}

    for approach, files in APPROACH_FILES.items():
        for file_name in files:
            for item in load_json(eval_dir / file_name):
                key = f"{approach}::{item['id']}"
                examples_by_key.setdefault(
                    key,
                    {
                        "approach": approach,
                        "id": item["id"],
                        "sentence_id": item.get("sentence_id"),
                        "sentence": item.get("sentence", ""),
                        "label": item.get("label", ""),
                        "label_description_full": item.get("label_description_full", ""),
                    },
                )

    return [examples_by_key[key] for key in sorted(examples_by_key)]


def build_prompt(example: dict) -> str:
    return f"""You are evaluating whether a recovered MITRE ATT&CK technique label is a good match for a sentence.

Return exactly one judgment:
- plausible: the sentence clearly supports the assigned label
- mild: the sentence is somewhat related, but the match is weak, incomplete, or ambiguous
- negative: the sentence does not support the assigned label or points to a different behavior

Evaluate the following example.

Approach: {example['approach']}
Example ID: {example['id']}
Sentence ID: {example.get('sentence_id', '')}
Sentence: {example['sentence']}
Assigned label: {example['label']}
Label description: {example.get('label_description_full', '')}

Respond with JSON only in this format:
{{"judgment": "plausible|mild|negative", "reason": "one short sentence"}}"""


def extract_json_object(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in model response: {text!r}")
    return json.loads(text[start:end + 1])


def request_judgment(client: OpenAI, example: dict, model: str):
    response = client.responses.create(
        model=model,
        input=build_prompt(example),
    )
    payload = extract_json_object(response.output_text)
    judgment = payload.get("judgment")

    if judgment not in JUDGMENTS:
        raise ValueError(f"Invalid judgment {judgment!r} for {example['id']}")

    return {
        "judgment": judgment,
        "reason": payload.get("reason", "").strip(),
        "model": model,
    }


def main():
    eval_dir = get_eval_dir()
    cache_path = eval_dir / "llm_judgments_cache.json"
    model = DEFAULT_MODEL

    if cache_path.exists():
        cache = load_json(cache_path)
        if not isinstance(cache, dict):
            raise ValueError(f"Expected dict-shaped cache in {cache_path}")
    else:
        cache = {}

    examples = build_examples(eval_dir)
    client = OpenAI()

    missing = [example for example in examples if cache.get(f"{example['approach']}::{example['id']}", {}).get("judgment") not in JUDGMENTS]
    print(f"Using model: {model}")
    print(f"Examples total: {len(examples)}")
    print(f"Examples to generate: {len(missing)}")

    for idx, example in enumerate(missing, start=1):
        key = f"{example['approach']}::{example['id']}"
        cache[key] = request_judgment(client, example, model)

        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        print(f"[{idx}/{len(missing)}] {key}: {cache[key]['judgment']}")
        time.sleep(0.2)

    print(f"Saved cache to {cache_path.resolve()}")


if __name__ == "__main__":
    main()
