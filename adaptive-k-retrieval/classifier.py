import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Dict, Literal
import time

load_dotenv()
client = openai.OpenAI()


SIMPLE_PROMPT = "You are a query classifier. If you can already answer the query, give me your answer. If you need more information to answer the query, you should classify it as `insufficient`."
HYBRID_PROMPT = (
    "Your task is to classify the given query. "
    "If you can answer the query, provide the answer. "
    "If you need more information to answer the query, answer either `few-shot' or `many-shot' depending on the amount of information you need. "
    "If you only need a few sentences to answer the query, classify it as `few-shot`. "
    "If you need a lot of information to answer the query, classify it as `many-shot`. "
)


class ClassifierAnswer(BaseModel):
    answer: str


def classify(query: str,
             model: str = "gpt-4o-mini",
             mode: Literal["binary", "hybrid"] = "binary",
             prompt_template: str = None,
             structured_output: bool = False,
             temperature: float = 0.0) -> Dict[str, str | int]:
    """Classify a query using an LLM."""
    if prompt_template is None:
        if mode == "binary":
            system_prompt = SIMPLE_PROMPT
        elif mode == "hybrid":
            system_prompt = HYBRID_PROMPT
    else:
        system_prompt = prompt_template 
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    
    t = time.time()
    if structured_output:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ClassifierAnswer,
            temperature=temperature,
        )
        category = response.choices[0].message.parsed.model_dump()["answer"]
        
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        category = response.choices[0].message.content
    reasoning_time = time.time() - t
    print(f"Classification time: {reasoning_time:.2f} seconds")
        
    num_input_tokens: int = response.usage.prompt_tokens
    num_output_tokens: int = response.usage.completion_tokens
        
    return {"category": category,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "reasoning_time": reasoning_time,}
            


def categorize(response: str):
    """Categorize the response."""
    if "insufficient" in response.lower():
        return "insufficient"
    else:
        return response


if __name__ == "__main__":
    query = "What is the capital of France?"
    result = classify(query)
    result = categorize(result)
    print(result)
