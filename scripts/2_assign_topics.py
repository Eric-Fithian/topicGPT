from topicgpt_python import (
    assign_topics,
)

import yaml
from pathlib import Path

configs = []
for config_path in Path("experiment_configs").glob("*.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    configs.append(config)

# Lvl 2 topic assignment
for config in configs:
    assign_topics(
        api=config["api"]["api_type"],
        model=config["api"]["model"],
        data=config["data_sample"],
        prompt_file=config["assignment"]["prompt"],
        out_file=config["assignment"]["output_1"],
        topic_file=config["generation"]["topic_output"],
        verbose=config["verbose"],
        base_url=config["api"]["base_url"],
        api_key=config["api"]["api_key"],
    )
