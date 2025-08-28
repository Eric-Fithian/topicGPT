from topicgpt_python import (
    generate_topic_lvl1,
)

import yaml
from pathlib import Path

configs = []
for config_path in Path("experiment_configs").glob("*.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    configs.append(config)

for config in configs:
    generate_topic_lvl1(
        api=config["api"]["api_type"],
        model=config["api"]["model"],
        data=config["data_sample"],
        prompt_file=config["generation"]["prompt"],
        seed_file=config["generation"]["seed"],
        out_file=config["generation"]["output"],
        topic_file=config["generation"]["topic_output"],
        verbose=config["verbose"],
        base_url=config["api"]["base_url"],
        api_key=config["api"]["api_key"],
    )
