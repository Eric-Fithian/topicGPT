from pathlib import Path
import yaml

CONFIG_DIR = Path("experiment_configs")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_DIR = Path("prompt")
if not PROMPT_DIR.exists():
    raise FileNotFoundError(
        f"Prompt directory {PROMPT_DIR} not found. Must provide prompts in {PROMPT_DIR}."
    )
PROMPT_NAMES = [
    "generation_1.txt",
    # "generation_2.txt",
    "assignment.txt",
    # "correction.txt",
]
for prompt_name in PROMPT_NAMES:
    prompt_file = PROMPT_DIR / prompt_name
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"Prompt file {prompt_file} not found. Must provide prompts in {PROMPT_DIR}."
        )

DATA_PATH = Path("data/input/news_sample_25.jsonl")

SEED_DIR = Path("data/pre_generated_topic_lists/seeds")

configs: list[dict] = []
for seed_path in SEED_DIR.glob("*.md"):
    config_id = f"{DATA_PATH.stem}__{PROMPT_DIR}__{seed_path.stem}"
    output_dir = Path(f"data/output/{config_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {}
    config["config_id"] = config_id
    config["api"] = {
        "api_key": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "model": "openai/gpt-oss-20b",
        "api_type": "openai-compatible",
    }
    config["verbose"] = True
    config["data_sample"] = str(DATA_PATH)
    config["generation"] = {
        "prompt": str(PROMPT_DIR / "generation_1.txt"),
        "seed": str(seed_path),
        "output": str(output_dir / "gen_1_raw_llm_resp.jsonl"),
        "topic_output": str(output_dir / "gen_1_topic_list.md"),
        "topic_output_csv": str(output_dir / "gen_1_topic_list.csv"),
    }
    config["assignment"] = {
        "prompt": str(PROMPT_DIR / "assignment.txt"),
        "output_1": str(output_dir / "ass_1_raw_llm_resp.jsonl"),
        "topic_output_csv": str(output_dir / "ass_1_topic_assignments.csv"),
    }
    configs.append(config)

for config in configs:
    with open(CONFIG_DIR / f"config__{config["config_id"]}.yml", "w") as f:
        yaml.dump(config, f)
