#%%
import pandas as pd
import argparse
import json

#%%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cnn")

    return parser.parse_args()

args = parse_args()
#%%
# class Args(argparse.Namespace):
#   dataset = 'cnn'

# args=Args()

if args.dataset == "cnn":
    df = pd.read_csv(
        "data/raw/cnn/cnn-test.csv",
        names=["_", "Article", "Summary"],
        usecols=["Article", "Summary"],
        header=0
    )

if args.dataset == "duc":
    # TODO: add the logic for parsing the TAC/DUC dataset
    raise NotImplementedError

# select a subset
SAMPLE_SIZE = 1
df = df.head(SAMPLE_SIZE)

#%%
data = []
for index, row in df.iterrows():
    data.append(
        {
            "id" : index,
            "document": row["Article"],
            "human_summary" : {
                "content": row["Summary"]
            }
        }
    )


#%%
from pathlib import Path
output_path = f"data/input/{args.dataset}/"
Path(output_path).mkdir(parents=True, exist_ok=True)

# %%

with open(f"{output_path}/{args.dataset}.json", "w") as outfile:
    json.dump(data, outfile, indent=4)