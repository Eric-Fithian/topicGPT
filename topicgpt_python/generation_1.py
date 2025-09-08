import argparse
import os
import traceback

import pandas as pd
import regex
from tqdm import tqdm

from topicgpt_python.utils import *  # noqa: F403

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _maybe_get_sbert():
    """Lazily import SentenceTransformer to avoid heavy deps at import-time."""
    try:
        if "TRANSFORMERS_NO_TORCHVISION" not in os.environ:
            os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
        from sentence_transformers import SentenceTransformer, util as st_util

        return SentenceTransformer("all-MiniLM-L6-v2"), st_util
    except Exception:
        return None, None


def prompt_formatting(
    generation_prompt,
    api_client,
    doc,
    seed_file,
    topics_list,
    context_len,
    verbose,
    max_top_len=500,  # Maximum length of topics
):
    """
    Format prompt to include document and seed topics.
    Handle cases where prompt is too long.
    """
    # Only pass the label (before ':') for seeds to keep prompt compact
    topic_str = "\n".join([topic.split(":")[0].strip() for topic in topics_list])

    # Calculate length of document, seed topics, and prompt
    doc_len = api_client.estimate_token_count(doc)
    prompt_len = api_client.estimate_token_count(generation_prompt)
    topic_len = api_client.estimate_token_count(topic_str)
    total_len = prompt_len + doc_len + topic_len

    # Handle cases where prompt is too long
    if total_len > context_len:
        if doc_len > (context_len - prompt_len - max_top_len):  # Truncate document
            if verbose:
                print(f"Document is too long ({doc_len} tokens). Truncating...")
            doc = api_client.truncating(doc, context_len - prompt_len - max_top_len)
            prompt = generation_prompt.format(Document=doc, Topics=topic_str)
        else:  # Truncate topic list
            if verbose:
                print(f"Too many topics ({topic_len} tokens). Pruning...")
            sbert, st_util = _maybe_get_sbert()
            if sbert is not None:
                cos_sim = {}
                doc_emb = sbert.encode(doc, convert_to_tensor=True)
                for top in topics_list:
                    top_emb = sbert.encode(top, convert_to_tensor=True)
                    cos_sim[top] = st_util.cos_sim(top_emb, doc_emb)
                # sort by cosine similarity (descending)
                sim_topics = sorted(cos_sim, key=cos_sim.get, reverse=True)

                # Retain only similar topics that fit within the context length
                max_top_len_actual = context_len - prompt_len - doc_len
                seed_len, seed_str = 0, ""
                while seed_len < max_top_len_actual and sim_topics:
                    new_seed = sim_topics.pop(0)
                    token_count = api_client.estimate_token_count(new_seed + "\n")
                    if seed_len + token_count > max_top_len_actual:
                        break
                    seed_str += new_seed + "\n"
                    seed_len += token_count
                prompt = generation_prompt.format(Document=doc, Topics=seed_str)
            else:
                # Fallback: greedy pack topics until they fit
                max_top_len_actual = context_len - prompt_len - doc_len
                seed_len, seed_str = 0, ""
                for top in topics_list:
                    token_count = api_client.estimate_token_count(top + "\n")
                    if seed_len + token_count > max_top_len_actual:
                        break
                    seed_str += top + "\n"
                    seed_len += token_count
                prompt = generation_prompt.format(Document=doc, Topics=seed_str)
    else:
        prompt = generation_prompt.format(Document=doc, Topics=topic_str)

    return prompt


# Robust topic line regex:
# - multi-line, optional leading bullets/whitespace
# - [level]  label : description
TOPIC_RE = regex.compile(
    r"""
    (?m)                    # multi-line mode
    ^\s*                    # leading spaces
    (?:[-*]\s*)?            # optional list bullet
    \[(\d+)\]\s*            # [level]
    ([^:\n\[\]]+?)\s*       # label (anything up to colon; exclude brackets/newlines/colon)
    :\s*
    ([^\n]+?)\s*            # description (rest of the line)
    $                       # end of line
    """,
    regex.VERBOSE | regex.UNICODE,
)


def parse_topics(response: str):
    """Return list of (level:int, label:str, desc:str) parsed from a model response."""
    out = []
    for lvl_s, name, desc in TOPIC_RE.findall(response):
        lvl = int(lvl_s)
        name = name.strip()
        desc = desc.strip()
        if name and desc:
            out.append((lvl, name, desc))
    return out


def generate_topics(
    topics_root,
    topics_list,
    context_len,
    docs,
    seed_file,
    api_client,
    generation_prompt,
    temperature,
    max_tokens,
    top_p,
    verbose,
    early_stop=50,  # Modify this parameter to control early stopping
):
    """
    Generate topics from documents using LLMs.
    """
    responses = []
    docs_since_new_topic = 0

    for i, doc in enumerate(tqdm(docs)):
        prompt = prompt_formatting(
            generation_prompt,
            api_client,
            doc,
            seed_file,
            topics_list,
            context_len,
            verbose,
        )

        try:
            added_new_topic = False
            response = api_client.iterative_prompt(  # noqa: F405
                prompt, max_tokens, temperature, top_p=top_p, verbose=verbose
            )

            # Parse all valid topic lines from the whole response
            matches = parse_topics(response)

            if not matches and verbose:
                print(f"No parsable topics in response:\n{response}\n")

            for lvl, name, desc in matches:
                if lvl != 1:
                    if verbose:
                        print(
                            f"Lower level topics are not allowed: [{lvl}] {name}: {desc}. Skipping..."
                        )
                    continue

                dups = topics_root.find_duplicates(name, lvl)
                if dups:
                    dups[0].count += 1
                else:
                    topics_root._add_node(lvl, name, 1, desc, topics_root.root)
                    topics_list = topics_root.to_topic_list(desc=False, count=False)
                    added_new_topic = True

            if added_new_topic:
                docs_since_new_topic = 0
            else:
                docs_since_new_topic += 1
                if verbose:
                    print(f"Docs since new topic: {docs_since_new_topic}/{early_stop}")
            if docs_since_new_topic >= early_stop:
                if verbose:
                    print(
                        f"Early stop triggered: {docs_since_new_topic} consecutive documents without new topics."
                    )
                return responses, topics_list, topics_root

            if verbose:
                print(f"Topics: {response}")
                print("--------------------")
            responses.append(response)

        except KeyboardInterrupt:
            # Preserve progress cleanly on manual interrupt
            responses.append("Interrupted")
            break
        except Exception:
            traceback.print_exc()
            responses.append("Error")
            break

    return responses, topics_list, topics_root


def generate_topic_lvl1(
    api,
    model,
    data,
    prompt_file,
    seed_file,
    out_file,
    topic_file,
    verbose,
    base_url=None,
    api_key=None,
    early_stop=50,
):
    """
    Generate high-level topics
    Returns:
    - topics_root (TopicTree): Root node of the topic tree
    """
    api_client = APIClient(
        api=api, model=model, base_url=base_url, api_key=api_key
    )  # noqa: F405
    max_tokens, temperature, top_p = 1000, 0.0, 1.0

    if verbose:
        print("-------------------")
        print("Initializing topic generation...")
        print(f"Model: {model}")
        print(f"Data file: {data}")
        print(f"Prompt file: {prompt_file}")
        print(f"Seed file: {seed_file}")
        print(f"Output file: {out_file}")
        print(f"Topic file: {topic_file}")
        print("-------------------")

    # Model configuration
    context = (
        128000
        if model not in ["gpt-3.5-turbo", "gpt-4"]
        else (4096 if model == "gpt-3.5-turbo" else 8000)
    )
    context_len = context - max_tokens

    # Load data
    df = pd.read_json(data, lines=True)
    docs = df["text"].tolist()
    with open(prompt_file, "r") as f:
        generation_prompt = f.read()
    topics_root = TopicTree().from_seed_file(seed_file)  # noqa: F405
    topics_list = topics_root.to_topic_list(desc=True, count=False)

    # Generate topics
    responses, topics_list, topics_root = generate_topics(
        topics_root,
        topics_list,
        context_len,
        docs,
        seed_file,
        api_client,
        generation_prompt,
        temperature,
        max_tokens,
        top_p,
        verbose,
        early_stop,
    )

    # Save generated topics
    topics_root.to_file(topic_file)

    try:
        df = df.iloc[: len(responses)]
        df["responses"] = responses
        df.to_json(out_file, lines=True, orient="records")
    except Exception:
        traceback.print_exc()
        # Keep a plain-text backup if JSONL write fails
        backup_path = f"data/output/generation_1_backup_{model}.txt"
        with open(backup_path, "w") as f:
            for line in responses:
                print(line, file=f)

    return topics_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api",
        type=str,
        help="API to use ('openai', 'vertex', 'vllm', 'azure', 'gemini')",
        default="openai",
    )
    parser.add_argument("--model", type=str, help="Model to use", default="gpt-4")

    parser.add_argument(
        "--data",
        type=str,
        default="data/input/sample.jsonl",
        help="Data to run generation on",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompt/generation_1.txt",
        help="File to read prompts from",
    )
    parser.add_argument(
        "--seed_file",
        type=str,
        default="prompt/seed_1.md",
        help="Markdown file to read seed topics from",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="data/output/generation_1.jsonl",
        help="File to write results to",
    )
    parser.add_argument(
        "--topic_file",
        type=str,
        default="data/output/generation_1.md",
        help="File to write topics to",
    )

    parser.add_argument(
        "--verbose", type=bool, default=False, help="Whether to print out results"
    )
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    args = parser.parse_args()
    generate_topic_lvl1(
        args.api,
        args.model,
        args.data,
        args.prompt_file,
        args.seed_file,
        args.out_file,
        args.topic_file,
        args.verbose,
        args.base_url,
        args.api_key,
    )
