import pandas as pd

df = pd.read_csv("data/input/news.tsv", sep="\t", header=None)
columns = ["id", "topic", "subtopic", "title", "content", "url", "meta1", "meta2"]
df.columns = columns
df.to_csv("data/input/news.csv", index=False)
