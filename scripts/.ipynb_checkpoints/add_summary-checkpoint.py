#%%
import pandas as pd
import argparse
import json
import sys
from typing import Any, Dict, List
import logging 
from openai import OpenAI
import copy 
from pathlib import Path
import time 

# %env LAS_API_TOKEN=

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
        self.model = "gpt-3.5-turbo"
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

class ComposableAgent:
    """
    This class enables a series of iterative conversations to take place between the user 
    and the LLM. This can be used to provide few shot tuning to the LLM
    """
    def __init__(self, prompt_setup: List[str]) -> None:
        logging.basicConfig(filename="logs/.test",
                    filemode='w',
                    format='%(message)s',
                    level=logging.WARN)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("LAS_API_TOKEN")
        )
        self.model = "gpt-3.5-turbo"
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
        self.count = 1
    def chat(self, message: str, mute=False) -> str:
        if self.count<2:
            self.messages.append({"role": "user", "content": message})
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages,
                        temperature = 1.5
                    )
                    break
                except:
                    time.sleep(0.1)
                    print("sleeping due to error")
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        else:
            while True:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages+[{"role": "user", "content": message}],
                        temperature=0
                    )
                    break
                except:
                    time.sleep(0.1)
                    print("sleeping due to error")
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})  # potentially cull 
        result = response.choices[0].message.content

        self.logger.warning(result)
        self.count += 1
        return result

#%%
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cnn")
    parser.add_argument("--initialise", type=bool, default=True)
    parser.add_argument("--run_name", type=str, default="both")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--overwrite", type=bool, default=False) # whether to overwrite the current values
    parser.add_argument("--agent_structure", type=str, default="concensus")
    return parser.parse_args()

args = parse_args()

#%%
# class Args(argparse.Namespace):
#   dataset = 'cnn'
#   run_name = "both"
#   num_agents = 2
#   overwrite = False
#   initialise = True
#   agent_structure = "consensus"

# args=Args()
#%%
def enrich_json(task: str, json: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if "single" in task:
        summariser = PromptTunedAgent(
            [
                "You are a writer who is an expert is creating short summaries of news articles. The objective of these summaries is to capture all the important information from the original article in one sentence.Format the summary in a single paragraph.\nDo you understand your role?",
            ]
        )
        for row in json:
            row[task] = {"content" : summariser.chat(f'Summarise this article:\n{row["document"]}.')}
        return json
    elif "multi" in task:
        for row in json:
            row[task] = {
                "config" : {
                    "agentic_structure": args.agent_structure,
                    "number_of_agents": args.num_agents
                },
                "results": create_consensus_summaries(row["document"], num_agents=args.num_agents, iterations=4)
            }
    else:
        raise NotImplementedError
    
    return json
    
def create_consensus_summaries(article: str, num_agents: int=2, iterations: int=7) -> List[List[Dict[str,str]]]:
    iteration_results = []
    BASE_PROMPT = "You are a writer who is an expert is creating short summaries of news articles. The objective of these summaries is to capture all the important information from the original article in one sentence. Format the summary in a single paragraph.\nDo you understand your role?"
    FIRST_PROMPT = f"Summarise this article:\n{article}."
    agents = [ComposableAgent([BASE_PROMPT]) for i in range(num_agents)]
    for i in range(iterations):
        if i==0:
            agent_results = []
            for agent in agents:
                agent_results.append({"content": agent.chat(FIRST_PROMPT)})
            iteration_results.append(agent_results)
        else:
            agent_results = []
            for idx, agent in enumerate(agents):
                summary_list = copy.deepcopy(iteration_results[-1])
                summary_list.pop(idx)
                summary_list = [row["content"] for row in summary_list]
                iter_prompt = "These are the summaries from the other agents:"
                for summary in summary_list:
                    iter_prompt += f"\nAgent response: {summary}"
                iter_prompt += "\n\nProduce an updated summary which incorporates the best parts of the summaries from the other agents and preserves the best part of your current summary. In evaluating which parts of a summary are good you may consider the contents of the original article. The objective of these summaries is to capture all the important information from the original article in one sentence. Format the summary in a single paragraph. Don't give any explanation, just return the updated summary."
                agent_results.append({"content": agent.chat(iter_prompt)})
            iteration_results.append(agent_results)           

    return iteration_results

#%%
if args.initialise:
    input_path = f"data/input/{args.dataset}"
    with open(f"{input_path}/{args.dataset}.json", "r") as infile:
        input = json.loads(infile.read())
else:
    input_path = f"data/enriched/{args.dataset}"
    Path(input_path).mkdir(parents=True, exist_ok=True)
    with open(f"{input_path}/{args.dataset}.json", "r") as infile:
        input = json.loads(infile.read())
#%%
if args.run_name == "singleagent":
    tasks = [f"{args.run_name}_summary"]
elif args.run_name == "multiagent":
    tasks = [f"{args.run_name}_summary"]
elif args.run_name == "both":
    tasks = ["singleagent_summary", "multiagent_summary"]
else:
    raise NotImplementedError

for task in tasks:
    if (task in input[0].keys()) and (args.overwrite == False):
        continue
    else:
        output = enrich_json(task, input, args)

#%%
output_path = f"data/enriched/{args.dataset}"
Path(output_path).mkdir(parents=True, exist_ok=True)

#%%
with open(f"{output_path}/{args.dataset}.json", "w") as outfile:
    json.dump(output, outfile, indent=4)
