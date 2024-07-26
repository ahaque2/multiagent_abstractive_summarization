from openai import OpenAI
import os
from typing import List
import json
import logging


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
            messages=self.messages+[{"role": "user", "content": message}]
        )
        result = response.choices[0].message.content

        self.logger.warning(message)
        self.logger.warning(result)
        return result
    

def get_writer_critic_summary(docs: List[str], num_sentences: int=5, num_iterations: int=5):

    starting_summariser = PromptTunedAgent(
        [
            "You are a writer who is an expert is creating short summaries of news articles.The objective of these summaries is to capture all the important information from the origonal article in five sentences.Format the summary in a single paragraph.\nDo you understand your role?",
        ]
    )

    critic = PromptTunedAgent(
        [
            "You will be given a set of articles and a summary of the articles. Your task is to make a list of  any pieces of information from the article which are not contained in the summary. Then select the most important piece of information from this list. If this piece of information is more important than the information in the summary provide feedback explaining this piece of information. If it isn't more important than information in the summary then say 'Quality Control is Complete'. Do you understand the task?"
        ]
    )

    improver = PromptTunedAgent(
        [
            "You are a writer who is an expert in editing and improving short summaries of news articles. Your objective is to create summaries that capture important information from the origonal article in five sentences.\nDo you understand your role?",
            "In a moment you will be given a the article, a draft summary and some feedback about how to improve the draft summary. Using this information you will then produce a new summary. Specifically your should:\n\t - Consider whether the information included in the feedback is more important than the information in your current summary. If it is then you should try to include it. If not then ignore the feedack. \n\t - Your should create the new summary by making small changes to your original summary. Don't provide any formatting like 'Revised Summary:' \Do you understand?"
        ]
    )

    articles = "Article:\n\n".join(docs)
    summary = starting_summariser.chat(f"Summarise these articles:\n{articles}.")
    feedback = critic.chat(f"Here is the article and the summary. \nArticle:\n{articles}\nSummary:\n{summary}")
    improved_summary = improver.chat(f"The article, draft summary and feedback are as follows:\nArticle: {articles}\nDraft Summary: {summary}\nFeedback: {feedback}")
    for i in range(num_iterations-1):
            feedback = critic.chat(f"Here is the article and the summary. \nArticle:\n{articles}\nSummary:\n{improved_summary}")
            improved_summary = improver.chat(f"The article, draft summary and feedback are as follows:\nArticle: {articles}\nDraft Summary: {improved_summary}\nFeedback: {feedback}")
    return improved_summary


if __name__ == "__main__":
    with open("data/article.txt") as f:
        article = f.read()

    starting_summariser = PromptTunedAgent(
        [
            "You are a writer who is an expert is creating short summaries of news articles.The objective of these summaries is to capture all the important information from the origonal article in five sentences.Format the summary in a single paragraph.\nDo you understand your role?",
        ]
    )

    summary = starting_summariser.chat(f"Summarise this article:\n{article}.")

    critic = PromptTunedAgent(
        [
            "You will be given an article and a summary of the article. Your task is to make a list of  any pieces of information from the article which are not contained in the summary. Then select the most important piece of information from this list. If this piece of information is more important than the information in the summary provide feedback explaining this piece of information. If it isn't more important than information in the summary then say 'Quality Control is Complete'. Do you understand the task?"
        ]
    )

    feedback = critic.chat(f"Here is the article and the summary. \nArticle:\n{article}\nSummary:\n{summary}")

    improver = PromptTunedAgent(
        [
            "You are a writer who is an expert in editing and improving short summaries of news articles. Your objective is to create summaries that capture important information from the origonal article in five sentences.\nDo you understand your role?",
            "In a moment you will be given a the article, a draft summary and some feedback about how to improve the draft summary. Using this information you will then produce a new summary. Specifically your should:\n\t - Consider whether the information included in the feedback is more important than the information in your current summary. If it is then you should try to include it. If not then ignore the feedack. \n\t - Your should create the new summary by making small changes to your original summary. Don't provide any formatting like 'Revised Summary:' \Do you understand?"
        ]
    )


    improved_summary = improver.chat(f"The article, draft summary and feedback are as follows:\nArticle: {article}\nDraft Summary: {summary}\nFeedback: {feedback}")