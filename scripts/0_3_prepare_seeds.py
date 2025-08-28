import pandas as pd

df = pd.read_excel(
    "data/pre_generated_topic_lists/raw/IPTC-MediaTopic-NewsCodes.xlsx", header=1
)

lvl1_topics_df_raw = df.dropna(subset=["Level1/NewsCode"])

lvl1_topics_df = lvl1_topics_df_raw[
    [
        "Level1/NewsCode",
        "Name (en-US)",
        "Definition (en-US)",
        "SubjectCode mapping",
        "Wikidata mapping",
        "RetiredDate",
    ]
]

column_renaming = {
    "Name (en-US)": "topic_name",
    "Definition (en-US)": "topic_description",
}
lvl1_topics_df = lvl1_topics_df.rename(columns=column_renaming)


# save to csv
lvl1_topics_df.to_csv(
    "data/pre_generated_topic_lists/intermediate/IPTC_lvl1.csv", index=False
)


seed_sizes = [2, "all"]
for seed_size in seed_sizes:
    if seed_size == "all":
        seed_size = len(lvl1_topics_df)
    else:
        seed_size = int(seed_size)

    # Save to seed topic markdown format
    output_file = f"data/pre_generated_topic_lists/seeds/IPTC_seed_{seed_size}.md"
    with open(output_file, "w") as f:
        for row in lvl1_topics_df[:seed_size].itertuples():
            topic_name = row.topic_name.strip().lower()
            topic_description = row.topic_description.strip().lower()

            f.write(f"[1] {topic_name}: {topic_description}\n")
