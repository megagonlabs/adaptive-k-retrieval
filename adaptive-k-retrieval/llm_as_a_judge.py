import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import json
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm

# annotation
from typing import List, Dict, Literal, Optional, Any

load_dotenv()


class ParsedResponse(BaseModel):
    reasoning: str
    final_answer: str
    

class Eval(BaseModel):
    id: int
    label: Literal["Exact Match", "Partial Match", "No Match"]
    

class EvaluationResponse(BaseModel):
    evaluations: List[Eval]
    

EvaluationResponseType = List[Dict[str, int | str]]
LAAJEvaluationType = Dict[str, str | List[dict] | EvaluationResponseType]
    

async def structuralize(question: str,
                        prediction: str,
                        gold: str,
                        eval_template: str,
                        post_proc_template: str,
                        client: openai.OpenAI,
                        model_name: str = "gpt-4o-mini-2024-07-18",
                        k: int = 50) -> Dict[str, str | List[dict] | EvaluationResponseType]:
    """
    Processes a single row of the input data.
    This function handles both post-processing of the model's response
    and evaluation the prediction against the label.
    This is necessary to ensure that the model's response is in the correct format,
    since some LLMs do not have structured-output capabilities.
    
    Args:
        question (str): The input query.
        prediction (str): The model's prediction.
        gold (List[dict]): The ground truth labels.
        eval_template (str): The template for evaluation.
        post_proc_template (str): The template for post-processing.
        client (openai.OpenAI): The OpenAI API client.
        model_name (str): The name of the model to use.
        k (int): The maximum number of labels to process.
        
    Returns:
        dict: A dictionary containing:
            - question (str)
            - prediction (str)
            - gold (str): a json string; to be json.loads-ed later
            - evaluations (List[str, int | str]),
            - reasoning (str)
            - final_answer (str)
    """
    content = post_proc_template.format(response=prediction)
    
    # Post-process the model's response to get the final answer and reasoning
    try:
        response = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "user", "content": content}
            ],
            response_format=ParsedResponse,
            temperature=0.0,
            max_tokens=16384,
        )
    except openai.LengthFinishReasonError as e:
        print(f"Error: {e}")
        print("The final answer and the reasoning are not available and left empty. Evaluations are placeholders.")
        return {
            "question": question,
            "prediction": prediction,
            "gold": gold,
            "evaluations": [{"id": 0, "label": "No Match"}],
            "reasoning": "",
            "final_answer": "",
        }
    # print(response)
    parsed = response.choices[0].message.parsed.model_dump()
    print(gold)
    # print(type(gold))
    gold = json.loads(gold)
    gold = [{"id": i} | x for i, x in enumerate(gold[:k], 1)]
    
    if gold:
        output_format = [{"id": x["id"], "label": "..."} for x in gold]
    else:
        output_format = [{"id": 0, "label": "..."}]
        
    content = eval_template.format(question=question,
                                   pred=parsed["final_answer"],
                                   gold=gold,
                                   output_format=output_format)
    
    # Evaluate the model's prediction against the gold answer
    response = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {"role": "user", "content": content}
        ],
        response_format=EvaluationResponse,
        temperature=0.0,
    )
    evals = response.choices[0].message.parsed.model_dump()["evaluations"]
    
    return {
        "question": question,
        "prediction": prediction,
        "gold": gold,
        "evaluations": evals,
        "reasoning": parsed["reasoning"],
        "final_answer": parsed["final_answer"],
    }
    

async def evaluate(pred_file: str | Dict[str, Any] | List[Dict[str, Any]],
                   output_file: str,
                   eval_prompt_file: str,
                   post_process_prompt_file: str,
                   model_name: str = "gpt-4o-mini-2024-07-18",
                   k: int = 50,
                   api_key: str = None,
                   api_base: str = None) -> None:
    """
    Evaluates the model's predictions against the gold standard data.

    Args:
    - pred_file (str): Path to the prediction file, or dict of prediction results.
    - output_dir (str): Path to the output directory.
    - eval_prompt_file (str): Path to the evaluation template.
    - post_process_prompt_file (str): Path to the post-processing template.
    - model_name (str): Name of the OpenAI model to use.
    - api_key (str): OpenAI API key. If None, use the environment variable OPENAI_API_KEY.
        This is not necessary if the key is loaded through dotenv.
    - base_url (str): Base URL for the OpenAI API. Default is None.
        This is not necessary if the key is loaded through dotenv.
    """
    client = openai.AsyncOpenAI()
    
    if isinstance(pred_file, dict) or isinstance(pred_file, list):
        # WARNING: when we load_context in task_loader and store it in Dataset,
        # the DataFrame (gold) in a row is automatically converted to a dictionary.
        df = pd.DataFrame(pred_file)
        
    elif isinstance(pred_file, pd.DataFrame):
        df = pred_file
        
    else:
        df = pd.read_json(Path(pred_file))
            
    with open(eval_prompt_file) as f:
        eval_template = f.read()
    with open(post_process_prompt_file) as f:
        post_proc_template = f.read()
        
    tasks = [
        structuralize(
            question=row["question"],
            prediction=row["prediction"],
            gold=row["gold"],
            eval_template=eval_template,
            post_proc_template=post_proc_template,
            client=client,
            model_name=model_name,
            k=k
        )
        for _, row in df.iterrows()
    ]
    
    results = await tqdm.gather(*tasks)

    # Save the results
    df = pd.DataFrame(results).to_json(output_file,
                                  orient="records")
    print(f"Results saved to {output_file}")
    
    
def run(pred_file: str | dict,
        output_file: Optional[str],
        eval_prompt_file: str,
        post_process_prompt_file: str,
        model_name: str = "gpt-4o-mini-2024-07-18",
        k: int = 50) -> None:
    """Run the async evaluate function.
    
    Args:
        pred_file (str | dict): Path to the prediction file, or dict of prediction results
        output_file (str): Path to the output file.
        eval_prompt_file (str): Path to the evaluation prompt file.
        post_process_prompt_file (str): Path to the post-process prompt file.
        model_name (str): Name of the OpenAI model to use.
        k (int): Number of samples to evaluate.
    """
    asyncio.run(
        evaluate(
            pred_file=pred_file,
            output_file=output_file,
            eval_prompt_file=eval_prompt_file,
            post_process_prompt_file=post_process_prompt_file,
            model_name=model_name,
            k=k,
        )
    )
