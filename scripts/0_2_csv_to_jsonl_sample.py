import pandas as pd

df = pd.read_csv("data/input/news.csv")
n_sample = 10000
df = df.sample(n=n_sample)
# replace None with empty string
df["title"] = df["title"].fillna("")
df["content"] = df["content"].fillna("")
df["text"] = df["title"] + "\n\n" + df["content"]

# convert to jsonl and save
df[["id", "text"]].to_json(
    f"data/input/news_sample_{n_sample}.jsonl", orient="records", lines=True
)
