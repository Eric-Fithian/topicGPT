from pathlib import Path
import pandas as pd
import yaml
import re
import json

configs = []
for config_path in Path("experiment_configs").glob("*.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    configs.append(config)

for config in configs:
    # TOPIC OUTPUT
    topics = []
    with open(config["generation"]["topic_output"], "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            # Parse the line using regex to extract all components
            # Pattern: [1] Topic Name (Count: N): Description
            match = re.match(
                r"^\[(\d+)\]\s*(.*?)\s*\(Count:\s*(\d+)\)\s*:\s*(.*\S)\s*$", line
            )

            if match:
                topic_level = int(match.group(1))
                topic_name = str(match.group(2).strip())
                count = int(match.group(3))
                topic_description = str(match.group(4).strip())

                topic = {
                    "topic_level": topic_level,
                    "topic_name": topic_name,
                    "topic_description": topic_description,
                    "count": count,
                }
                topics.append(topic)

    # Create DataFrame and save to CSV
    df_topics = pd.DataFrame(topics)
    output_path = config["generation"]["topic_output_csv"]
    df_topics.to_csv(output_path, index=False)
    print(f"Saved topics to {output_path}")

    # TOPIC ASSIGNMENT
    assignments = []
    with open(config["assignment"]["output_1"], "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            try:
                data = json.loads(line)

                # Extract basic fields
                doc_id = str(data.get("id", ""))
                text = str(data.get("text", ""))
                responses = str(data.get("responses", ""))

                # Parse the responses field to extract topic information
                if (
                    responses
                    and responses
                    != "No topic from the provided hierarchy applies to this document."
                ):
                    # Extract topic level and name from responses
                    # Pattern: [1] Topic Name: Description (Supporting quote: "...")
                    topic_match = re.match(
                        r'(?s)^\[(\d+)\]\s*(.*?)\s*:\s*(.*?)(?:\s*(?:\(?\s*)?Supporting quote:\s*["“]([^"”]+)["”]\)?)?\s*$',
                        responses,
                    )

                    if topic_match:
                        topic_level = int(topic_match.group(1))
                        topic_name = str(topic_match.group(2).strip())
                        assignment_reason = str(topic_match.group(3).strip())
                        supporting_quote = str(
                            topic_match.group(4) if topic_match.group(4) else ""
                        )

                        assignment = {
                            "id": doc_id,
                            "text": text,
                            "topic_level": topic_level,
                            "topic_name": topic_name,
                            "assignment_reason": assignment_reason,
                            "supporting_quote": supporting_quote,
                        }
                        assignments.append(assignment)
                else:
                    # Handle cases with no topic assignment
                    assignment = {
                        "id": doc_id,
                        "text": text,
                        "topic_level": None,
                        "topic_name": None,
                        "assignment_reason": None,
                        "supporting_quote": None,
                    }
                    assignments.append(assignment)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue

    # Create DataFrame and save to CSV
    df_assignments = pd.DataFrame(assignments)
    output_path = config["assignment"]["topic_output_csv"]
    df_assignments.to_csv(output_path, index=False)
    print(f"Saved assignments to {output_path}")

    # Check that all topic names are in the topic output and their descriptions are the same
    for assignment in assignments:
        topic_name = assignment["topic_name"]
        if topic_name is None:
            continue
        if topic_name not in df_topics["topic_name"].values:
            raise ValueError(f"Topic name {topic_name} not found in topic output")
