from rouge import Rouge
from typing import List

def calculate_rouge(generated: List[str], reference: List[str]) -> List[float]:
    """
    Returns the f1 score for Rouge-LCS between two summaries
    """
    rouge = Rouge()
    scores = rouge.get_scores(generated, reference)
    return [score["rouge-l"]["f"] for score in scores]

if __name__ == "__main__":
    GENERATED = "Text summarization with Transformers is efficient for producing precise and relevant summaries."
    REFERENCE = "Text summarization with Transformers can be used to produce precise and relevant summaries."

    rouge = calculate_rouge(
        [GENERATED],
        [REFERENCE]
    )
    print(f"rouge = {rouge[0]}")
