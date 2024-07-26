#%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os
from multiagent_summariser import get_multiagent_summary
from writer_critic import get_writer_critic_summary
import json
import logging 
from openai import OpenAI
from typing import List
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

#%%
class PromptTunedAgent:
    """
    This class enables a series of iterative conversations to take place between the user 
    and the LLM. This can be used to provide few shot tuning to the LLM
    """
    def __init__(self, prompt_setup: List[str]) -> None:
        logging.basicConfig(filename="logs/.log",
                    filemode='w',
                    format='%(message)s',
                    level=logging.WARN)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("LAS_API_TOKEN")
        )
        self.model = "gpt-4o"
        self.temperature=1.0
        self.messages = [
        ]

        for message in prompt_setup:
            self.messages.append({"role": "user", "content": message})
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        
        self.logger.debug(json.dumps(self.messages, indent=2))

    def chat(self, message: str, mute=False) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages+[{"role": "user", "content": message}],
            temperature=self.temperature
        )
        result = response.choices[0].message.content

        self.logger.warning(result)
        return result

#%%
df = pd.read_csv("data/input/sample/sample_conflicting_news_dataset.csv")
df = df.head(2)
multi_agent_summary = []
writer_critic_summary = []
cluster = []
num_sentences = 5
# %%
for cluster_id in set(df["cluster_id"]):
    clustered_articles = df[df["cluster_id"]==cluster_id]
    docs = clustered_articles["text"]
    personalisation = ["Your summaries should be written from the perspertive a person with right leaning political tendancies"]*2
    personalisation += ["Your summaries should be written from the perspertive a person with left leaning political tendancies"]*2
    multi_agent_summary.append(get_multiagent_summary(docs, personalisation, num_sentences=num_sentences))
    articles = '\n'.join(docs)
    writer_critic_summary.append(get_writer_critic_summary(docs, num_sentences=num_sentences))
    cluster.append(cluster_id)

results_df = pd.DataFrame.from_dict({"cluster": cluster, "multiagent_summary": multi_agent_summary, "writer_critic_summary": writer_critic_summary})
results_df.to_csv("sample_conflicting_news_dataset_enriched.csv")