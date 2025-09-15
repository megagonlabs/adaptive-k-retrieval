import re
import string
from collections import Counter
import pandas as pd
from dotenv import load_dotenv
import json

# For annotation
from typing import Dict, List, Optional, Any
from collections.abc import Callable
from abc import ABC, abstractmethod
import argparse

# local
import llm_as_a_judge


# load environment variables, including OpenAI API key
load_dotenv()

TASKS = ["nq", "triviaqa", "hotpotqa", "popqa", "holobench", "all"]
QA_TASKS = ["nq", "triviaqa", "hotpotqa", "popqa"]

ContextType = Dict[str, str | int]
DocType = Dict[str, str | int | float]
RetrievalResultType = Dict[str, ContextType | List[DocType] | int | float]
GenerationResultType = Dict[str, str | int]
HoloBenchGenerationType = Dict[str, str | List[dict]
                               | List[float] | float]  # TODO: Specify the dict
HoloBenchGoldType = Dict[str, str | int | float]
ResultType = Dict[str, List[RetrievalResultType | GenerationResultType]]


class BaseEvaluator(ABC):
    """
    Base class for all evaluators.
    """
    @abstractmethod
    def compute_metrics(self, *args, **kwargs) -> dict:
        """
        Compute metrics for the given arguments.
        """
        ...


class RetrievalEvaluator(BaseEvaluator):
    """Evaluator class for retrieval.

    Largely based on RAGChecker.
    See also:
    - paper: https://arxiv.org/abs/2408.08067
    - code:  https://github.com/amazon-science/RAGChecker/blob/main/ragchecker/evaluator.py
    """
    def __init__(self,
                 self_route: bool = False,
                 full_context_generation_file: Optional[str] = None,
                 full_context_retrieval_file: Optional[str] = None):
        """
        Args:
            self_route (bool): Whether to use Self-ROUTE (Li et al. 2024).
            full_context_generation_file (Optional[str]): The path to the full context generation result file.
            full_context_retrieval_file (Optional[str]): The path to the full context retrieval result file.
        """
        self.self_route = self_route
        self.full_context_generation_file = full_context_generation_file
        self.full_context_retrieval_file = full_context_retrieval_file
        
    def compute_context_precision(self,
                                  rel_docs: List[ContextType],
                                  retrieved_docs: List[ContextType]) -> float:
        """Context precision is defined as follows:
        Rel_docs = {d_1, d_2, ..., d_n} are the docs of the relevant context in the retrieved context.
        Retrieved_docs = {d_1, d_2, ..., d_k} are the docs of the retrieved context.
        Precision = |Relevant_docs & Retrieved_docs| / |Retrieved_docs|
                  = |Relevant_docs| / k

        In implementation,
        - for QA tasks (HotpotQA, NQ, TriviaQA, PopQA), we can use sample["positive_ctxs"] as rel_docs.
        - for HoloBench, use sample["relevant_context"] as rel_docs.

        Returns:
            float: Context precision."""
        rel_doc_ids = [doc["psg_id"] for doc in rel_docs]
        retrieved_doc_ids = [doc["psg_id"] for doc in retrieved_docs]
        return len(set(rel_doc_ids) & set(retrieved_doc_ids)) / len(retrieved_doc_ids)

    def compute_context_recall(self,
                               rel_docs: List[ContextType],
                               retrieved_docs: List[ContextType]) -> float:
        """Context recall is defined as follows:
        Rel_docs = {d_1, d_2, ..., d_n} are the docs of the relevant context in the retrieved context.
        Retrieved_docs = {d_1, d_2, ..., d_k} are the docs of the retrieved context.
        Recall = |Relevant_docs & Retrieved_docs| / |Relevant_docs|
        """
        rel_doc_ids = [doc["psg_id"] for doc in rel_docs]
        retrieved_doc_ids = [doc["psg_id"] for doc in retrieved_docs]
        return len(set(rel_doc_ids) & set(retrieved_doc_ids)) / len(rel_doc_ids)
    
    def compute_context_f1(self,
                           precision: float,
                           recall: float) -> float:
        """Compute F1 score for context precision and recall.
        
        Args:
            precision (float): Context precision.
            recall (float): Context recall.
        Returns:
            float: The F1 score.
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_metrics(self,
                        retrieval_results: List[RetrievalResultType]) -> dict:
        """Evaluate retrieval.
        
        A retrieval result looks like:
        {"question": query,
        "sample_idx": sample["idx"],
        "retrieved_docs": retrieved_chunks,
        "retrieval_time": retrieval_time,
        "num_retrieved_docs": 0 if not retrieved_chunks else len(retrieved_chunks),
        "reduction_ratio": reduction_ratio,
        "true_k": true_k}
        
        Args:
            retrieval_results (List[RetrievalResultType]): The retrieval results.
                Each retrieval result is a dict with the following keys:
                    - "question": The query question.
                    - "sample_idx": The index of the sample.
                    - "retrieved_docs": The retrieved documents.
                    - "retrieval_time": The time taken for retrieval.
                    - "num_retrieved_docs": The number of retrieved documents.
                    - "reduction_ratio": The reduction ratio.
                    - "true_k": The true k value.
        Returns:
            dict: The computed metrics.
        """

        if self.self_route:
            with open(self.full_context_generation_file) as f:
                generation_result_full = json.load(f)["detail"]

            with open(self.full_context_retrieval_file) as f:
                retrieval_result_full = json.load(f)

        results = []
        
        for query_id, r_result in enumerate(retrieval_results):
            result = {}
            result.update(r_result)

            if self.self_route:
                if generation_result_full[query_id]["from_full_context"]:
                    result["num_retrieved_docs"] = int(result["num_retrieved_docs"]) + int(retrieval_result_full[query_id]["num_retrieved_docs"])
            
            # Absolute difference between the number of retrieved docs and the true k
            if r_result["retrieved_docs"]:
                result.update({"diff_k": abs(int(r_result["num_retrieved_docs"]) - int(r_result["true_k"]))})
            else:
                result.update({"diff_k": None}) # NA


            # Compute context precision and recall
            if r_result["retrieved_docs"]:
                context_precision = self.compute_context_precision(
                    r_result["positive_docs"],
                    r_result["retrieved_docs"],
                )
                context_recall = self.compute_context_recall(
                    r_result["positive_docs"],
                    r_result["retrieved_docs"]
                )
                context_f1 = self.compute_context_f1(
                    context_precision, context_recall
                )
            else:
                context_precision = 0.0
                context_recall = 0.0
                context_f1 = 0.0

            result.update({
                "context_precision": context_precision,
                "context_recall": context_recall,
                "context_f1": context_f1
            })
            results.append(result)
        
        overall_diff_k = [r["diff_k"] for r in results if r["diff_k"] is not None]
        overall_avg_diff_k = sum(overall_diff_k) / len(overall_diff_k) if len(overall_diff_k) else None
        overall_std_diff_k = None if overall_avg_diff_k is None else pd.Series(overall_diff_k).std()
            
        results = {
            "detail": results,
            "overall_avg_context_precision": sum([r["context_precision"] for r in results]) / len(results),
            "overall_std_context_precision": pd.Series([r["context_precision"] for r in results]).std(),
            "overall_avg_context_recall": sum([r["context_recall"] for r in results]) / len(results),
            "overall_std_context_recall": pd.Series([r["context_recall"] for r in results]).std(),
            "overall_avg_context_f1": sum([r["context_f1"] for r in results]) / len(results),
            "overall_std_context_f1": pd.Series([r["context_f1"] for r in results]).std(),
            "overall_avg_retrieval_time": sum([r["retrieval_time"] for r in results]) / len(results),
            "overall_std_retrieval_time": pd.Series([r["retrieval_time"] for r in results]).std(),
            "overall_avg_num_retrieved_docs": sum([int(r["num_retrieved_docs"]) for r in results]) / len(results),
            "overall_std_num_retrieved_docs": pd.Series([int(r["num_retrieved_docs"]) for r in results]).std(),
            "overall_avg_reduction_ratio": sum([r["reduction_ratio"] for r in results]) / len(results),
            "overall_std_reduction_ratio": pd.Series([r["reduction_ratio"] for r in results]).std(),
            "overall_avg_diff_k": overall_avg_diff_k,
            "overall_std_diff_k": overall_std_diff_k
        }
        
        return results


class QAGenerationEvaluator(BaseEvaluator):
    """
    Evaluator for question generation and answering tasks.
    Largely based on https://github.com/princeton-nlp/HELMET/blob/main/utils.py
    """

    def __init__(self, 
                 self_route: bool = False,
                 full_context_file: Optional[str] = None):
        """
        Args:
            self_route (bool): Whether to use Self-ROUTE (Li et al. 2024).
            full_context_file (Optional[str]): The path to the full context file.
        """
        self.self_route = self_route
        if self_route:
            self.result_full = pd.read_json(full_context_file)["detail"]
            print(f"Loading the full context file: {full_context_file}")

    def normalize_answer(self,
                         text: str) -> str:
        """Lowercase text and remove articles, punctuation, and extra whitespace.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        # Remove articles
        # TODO: make it language agnostic
        text = re.sub(r"\b(a|an|the)\b", "", text)

        # Fix whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation
        pattern = f"[{re.escape(string.punctuation)}]"
        text = re.sub(pattern, "", text)

        return text.lower()

    def exact_match(self,
                    prediction: str,
                    ground_truth: str) -> bool:
        """Check if the prediction matches the ground truth exactly.

        Args:
            prediction (str): The predicted text.
            ground_truth (str): The ground truth text.

        Returns:
            bool: True if the prediction matches the ground truth, False otherwise.
        """
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)

    def substring_exact_match(self,
                              prediction: str,
                              ground_truth: str) -> bool:
        """Check if the prediction is a substring of the ground truth.

        Args:
            prediction (str): The predicted text.
            ground_truth (str): The ground truth text.

        Returns:
            bool: True if the prediction is a substring of the ground truth, False otherwise.
        """
        return self.normalize_answer(ground_truth) in self.normalize_answer(prediction)

    def f1_score(self,
                 prediction: str,
                 ground_truth: str) -> Dict[str, float]:
        """Calculate the F1 score between the prediction and ground truth.

        Args:
            prediction (str): The predicted text.
            ground_truth (str): The ground truth text.

        Returns:
            Dict[str, float]: A dictionary containing precision, recall, and f1 score.
        """
        norm_pred = self.normalize_answer(prediction)
        norm_label = self.normalize_answer(ground_truth)

        zero_metric = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if norm_pred in ["yes", "no", "noanswer"] and norm_pred != norm_label:
            return zero_metric
        if norm_label in ["yes", "no", "noanswer"] and norm_pred != norm_label:
            return zero_metric

        pred_tokens = Counter(norm_pred.split())
        label_tokens = Counter(norm_label.split())
        common_tokens = pred_tokens & label_tokens
        num_common = sum(common_tokens.values())
        if num_common == 0:
            return zero_metric
        precision = num_common / len(pred_tokens)
        recall = num_common / len(label_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)

        return {"precision": precision,
                "recall": recall,
                "f1": f1}

    def max_over_labels(self,
                        metric_fn: Callable,
                        pred: str,
                        labels: List[str] | List[List[str]] | str) -> float:
        """Calculate the maximum score over multiple labels.
        Args:
            metric_fn (Callable): The metric function to use.
            pred (str): The predicted text.
            labels (List[str] | str): The ground truth labels.
        Returns:
            float: The maximum score.
        """
        if isinstance(labels, str):
            labels = [labels]
        elif isinstance(labels[0], list):
            # flatten to 1D list
            labels = [item for sublist in labels for item in sublist]

        scores = [metric_fn(pred, label) for label in labels]
        if isinstance(scores[0], dict):
            # for self.f1_score() method
            scores = [score["f1"] for score in scores]
        else:
            scores = [float(score) for score in scores]

        return max(scores)

    def compute_single_metrics(self,
                               prediction: str,
                               labels: List[str] | List[List[str]] | str) -> Dict[str, float]:
        """
        Compute metrics for question generation and answering tasks.
        
        Args:
            prediction (str): The predicted text.
            labels (List[str] | List[List[str]] | str): The ground truth labels.
        Returns:
            Dict[str, float]: A dictionary containing exact match, substring exact match, and f1 score.
        """
        em = self.max_over_labels(self.exact_match, prediction, labels)
        sub_em = self.max_over_labels(
            self.substring_exact_match, prediction, labels
        )
        f1 = self.max_over_labels(self.f1_score, prediction, labels)

        return {
            "exact_match": em,
            "substring_exact_match": sub_em,
            "f1_score": f1
        }

    def compute_metrics(self,
                        generation_results: List[GenerationResultType],
                        retrieval_results: List[RetrievalResultType]) -> Dict[str, Any]:
        """Compute metrics for question generation and answering tasks.
        
        Args:
            generation_results (List[GenerationResultType]): The generation results.
            retrieval_results (List[RetrievalResultType]): The retrieval results.
        Returns:
            Dict[str, Any]: The computed metrics.
        """
        results = []
        for i, (g_result, r_result) in enumerate(zip(generation_results, retrieval_results)):
            if self.self_route and "unanswerable" in g_result["prediction"]:
                result = {"from_full_context": True}
                self.result_full[i]["num_input_tokens"] = int(g_result["num_input_tokens"]) + int(self.result_full[i]["num_input_tokens"])
                self.result_full[i]["num_output_tokens"] = int(g_result["num_output_tokens"]) + int(self.result_full[i]["num_output_tokens"])
                self.result_full[i]["reasoning_time"] = float(g_result["reasoning_time"]) + float(self.result_full[i]["reasoning_time"])
                result.update(self.result_full[i])
            else:
                result = {"from_full_context": False}
                result.update(g_result)

                score = self.compute_single_metrics(
                    g_result["prediction"], g_result["gold"])
                # print(f"Score: {score}")

                result.update(score)
                result.update(r_result)
            results.append(result)

        results = {
            "detail": results,
            "overall_avg_f1_score": sum([r["f1_score"] for r in results]) / len(results),
            "overall_std_f1_score": pd.Series([r["f1_score"] for r in results]).std(),
            "overall_max_f1_score": max([r["f1_score"] for r in results]),
            "overall_min_f1_score": min([r["f1_score"] for r in results]),
            "overall_avg_exact_match": sum([r["exact_match"] for r in results]) / len(results),
            "overall_std_exact_match": pd.Series([r["exact_match"] for r in results]).std(),
            "overall_avg_substring_exact_match": sum([r["substring_exact_match"] for r in results]) / len(results),
            "overall_std_substring_exact_match": pd.Series([r["substring_exact_match"] for r in results]).std(),
            "overall_avg_reasoning_time": sum([r["reasoning_time"] for r in results]) / len(results),
            "overall_std_reasoning_time": pd.Series([r["reasoning_time"] for r in results]).std(),
            "overall_avg_num_input_tokens": sum([r["num_input_tokens"] for r in results]) / len(results),
            "overall_std_num_input_tokens": pd.Series([r["num_input_tokens"] for r in results]).std(),
            "overall_avg_num_output_tokens": sum([r["num_output_tokens"] for r in results]) / len(results),
            "overall_std_num_output_tokens": pd.Series([r["num_output_tokens"] for r in results]).std(),
        }

        return results


class HoloBenchGenerationEvaluator(BaseEvaluator):
    """Evaluator class for HoloBench tasks."""

    def __init__(self,
                 self_route: bool,
                 pred_file: str | dict,
                 output_file: Optional[str],
                 generation_eval_score_file: str,
                 eval_prompt_file: str,
                 post_process_prompt_file: str,
                 judge_model_name: str,
                 k: int):
        """
        Args:
            self_route (bool): Whether to use Self-ROUTE (Li et al. 2024).
            full_context_file (Optional[str]): The path to the full context file.
            pred_file (str | dict): The path to the prediction file or the prediction dict.
            output_file (Optional[str]): The path to the output file.
            generation_eval_score_file (str): The path to the generation evaluation score file.
            eval_prompt_file (str): The path to the evaluation prompt file.
            post_process_prompt_file (str): The path to the post-process prompt file.
            judge_model_name (str): The name of the judge model.
            k (int): The number of samples in the prediction to evaluate.
                See also the LLM-as-a-judge implementation in the original HoloBench paper.
        """
        self.self_route = self_route
        self.pred_file = pred_file
        self.output_file = output_file
        self.generation_eval_score_file = generation_eval_score_file
        self.eval_prompt_file = eval_prompt_file
        self.post_process_prompt_file = post_process_prompt_file
        self.judge_model_name = judge_model_name
        self.k = k

        if self.self_route: # load the full-context result
            full_context_file = self.generation_eval_score_file.replace("self-route", "full-context").replace("gte-Qwen2-1.5B-instruct", "bge-large-en-v1.5").replace("generation_eval", "generation_results")
            print(f"Loading full-context result from {full_context_file}")
            with open(full_context_file) as f:
                self.result_full = json.load(f)

    def run_llm_as_a_judge(self) -> str:
        """Run LLM-as-a-judge evaluation."""
        llm_as_a_judge.run(pred_file=self.pred_file,
                           output_file=self.output_file,
                           eval_prompt_file=self.eval_prompt_file,
                           post_process_prompt_file=self.post_process_prompt_file,
                           model_name=self.judge_model_name,
                           k=self.k)
        # The results will be saved in {output_dir}/{pred_file}
        # The resulting json file has the structure of: List[LAAJEvaluationType],
        # where LAAJEvaluationType = Dict[str, int | str]
        print(f"Evaluation results saved to {self.output_file}")

        return self.output_file

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute metrics.

        Args:
            query (str): The query.
            pred (str): The predicted text.
            gold (dict): The ground truth read from the json file.

        Returns:
            Dict[str, Any]: The computed metrics.
        """
        self.run_llm_as_a_judge()
        # Read the evaluation result file
        with open(self.output_file, "r") as f:
            results = json.load(f)  # -> List[dict]
        """
        result: List[LAAJEvaluationType],
        where LAAJEvaluationType = Dict[str, str | List[dict] | EvaluationResponseType]
        containing:
            - question (str)
            - prediction (str)
            - gold (List[dict])
            - evaluations (EvaluationResponseType = List[str, int | str]),
            - reasoning (str)
            - final_answer (str)
        """

        for i, result in enumerate(results):
            results[i]["evaluation_scores"] = [
                1 if e["label"] == "Exact Match"
                else 0.5 if e["label"] == "Partial Match"
                else 0
                for e in result["evaluations"]
            ]
            results[i]["avg_score"] = sum(
                result["evaluation_scores"]) / len(result["evaluation_scores"]
                                                )
            if "avg_score" not in results[i]:
                raise ValueError(f"avg_score not found in the result: {result}")
                
        # additional generation metrics
        if isinstance(self.pred_file, str):
            with open(self.pred_file) as f:
                generation_results = json.load(f)
        elif isinstance(self.pred_file, list):
            generation_results = self.pred_file
        else:
            raise ValueError("pred_file must be a str or list of dicts.")
        
        for i, gen_result in enumerate(generation_results):
            # update the generation result with the evaluation result
            if self.self_route: 
                if "unanswerable" in gen_result["prediction"]:
                    generation_results[i]["from_full_context"] = True
                    generation_results[i]["num_input_tokens"] = int(generation_results[i]["num_input_tokens"]) + int(self.result_full[i]["num_input_tokens"])
                    generation_results[i]["num_output_tokens"] = int(generation_results[i]["num_output_tokens"]) + int(self.result_full[i]["num_output_tokens"])
                    generation_results[i]["reasoning_time"] = float(generation_results[i]["reasoning_time"]) + float(self.result_full[i]["reasoning_time"])
                else:
                    generation_results[i]["from_full_context"] = False
                
                results[i].update(generation_results[i])
            
        results = {
            "detail": results,
            "overall_avg_score": sum([r["avg_score"] for r in results]) / len(results),
            "overall_std_score": pd.Series([r["avg_score"] for r in results]).std(),
            "overall_max_score": max([r["avg_score"] for r in results]),
            "overall_min_score": min([r["avg_score"] for r in results]),
            "overall_avg_reasoning_time": sum([r["reasoning_time"] for r in generation_results]) / len(generation_results),
            "overall_std_reasoning_time": pd.Series([r["reasoning_time"] for r in generation_results]).std(),
            "overall_avg_num_input_tokens": sum([r["num_input_tokens"] for r in generation_results]) / len(generation_results),
            "overall_std_num_input_tokens": pd.Series([r["num_input_tokens"] for r in generation_results]).std(),
            "overall_avg_num_output_tokens": sum([r["num_output_tokens"] for r in generation_results]) / len(generation_results),
            "overall_std_num_output_tokens": pd.Series([r["num_output_tokens"] for r in generation_results]).std(),
        }

        return results


def get_args() -> argparse.Namespace:
    """Argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate the results of the adaptive RAG pipeline."
    )

    general_group = parser.add_argument_group("General")
    general_group.add_argument(
        "--task",
        type=str,
        required=True,
        choices=TASKS,
        help="Task.",
    )
    general_group.add_argument(
        "--self_route",
        action="store_true",
        help="Use self-routing.",
    )
    general_group.add_argument(
        "--retrieval_result_file",
        type=str,
        default="debug_retrieval_results.json",
        help="Path to the retrieval result file.",
    )
    general_group.add_argument(
        "--generation_result_file",
        type=str,
        default="debug_generation_results.json",
        help="Path to the generation result file.",
    )
    general_group.add_argument( # TODO: Delete this argument; not used anymore
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    general_group.add_argument(
        "--retrieval_eval_output_file",
        type=str,
        default="retrieval_eval_results.json",
        help="Path to the retrieval evaluation output file.",
    )
    general_group.add_argument(
        "--generation_eval_score_file",
        type=str,
        default="generation_eval_scores.json",
        help="Path to the final score output file.",
    )

    holobench_group = parser.add_argument_group("HoloBench")
    holobench_group.add_argument(
        "--eval_prompt_file",
        type=str,
        default="templates/holobench_eval_template.txt",
        help="Path to the evaluation prompt file.",
    )
    holobench_group.add_argument(
        "--eval_output_file",
        type=str,
        default="holobench_eval_results.json",
        help="Path to the evaluation output file.",
    )
    holobench_group.add_argument(
        "--post_process_prompt_file",
        type=str,
        default="templates/holobench_postproc_template.txt",
        help="Path to the post-processing prompt file.",
    )
    holobench_group.add_argument(
        "--judge_model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Name of the judge model.",
    )
    holobench_group.add_argument(
        "--holobench_k",
        type=int,
        default=50,
        help="Number of retrieved documents for HoloBench tasks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print(f"Self-route: {args.self_route}")
    # print retriever
    print(f"Retrieval result file: {args.retrieval_result_file}")

    with open(args.retrieval_result_file) as f:
        retrieval_results = json.load(f)
    with open(args.generation_result_file) as f:
        generation_results = json.load(f)

    """
    retrieval_result = {
            "query": str,
            "sample_idx": int,
            "retrieved_docs": list[dict] | None,
            "retrieval_time": float,
            "num_retrieved_docs": int,
            "reduction_ratio": float,
            "true_k": int | None
        }
        generation_result = {
            "query": str,
            "category": str,
            "prediction": str,
            "gold": str,
            "num_input_tokens": int,
            "num_output_tokens": int,
            "reasoning_time": float,
        }
    """

    # Generation evaluation
    if args.task in QA_TASKS:
        evaluator = QAGenerationEvaluator(
            self_route=args.self_route,
            full_context_file=args.generation_eval_score_file.replace("self-route", "full-context").replace("gte-Qwen2-1.5B-instruct", "bge-large-en-v1.5")
        )
        results = evaluator.compute_metrics(
            generation_results=generation_results,
            retrieval_results=retrieval_results
        )
        
    elif args.task == "holobench":
        if "generation_eval" in args.generation_eval_score_file:
            eval_output_file = args.generation_eval_score_file.replace("generation_eval", "holobench_eval")
        else:
            print("Warning: generation_eval_score_file does not contain 'generation_eval'.")
            eval_output_file = args.generation_eval_score_file.replace(".json", "_holobench_eval.json")
        evaluator = HoloBenchGenerationEvaluator(
            self_route=args.self_route,
            pred_file=generation_results,
            output_file=eval_output_file,
            generation_eval_score_file=args.generation_eval_score_file,
            eval_prompt_file=args.eval_prompt_file,
            post_process_prompt_file=args.post_process_prompt_file,
            judge_model_name=args.judge_model_name,
            k=args.holobench_k
        )
        results = evaluator.compute_metrics()
    
    for key, value in results.items():
        if key == "detail":
            continue
    print("Generation evaluation done.")

    # Save the results (List[dict])
    print("Saving the generation evaluation results...")
    
    with open(args.generation_eval_score_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.generation_eval_score_file}")

    # Retrieval evaluation
    print("Full-context results:", args.generation_eval_score_file.replace("gte-Qwen2-1.5B-instruct", "bge-large-en-v1.5"))

    retrieval_evaluator = RetrievalEvaluator(
        self_route=args.self_route,
        full_context_generation_file=args.generation_eval_score_file.replace("gte-Qwen2-1.5B-instruct", "bge-large-en-v1.5"),
        full_context_retrieval_file=args.retrieval_result_file.replace("self-route", "full-context").replace("gte-Qwen2-1.5B-instruct", "bge-large-en-v1.5"))
    retrieval_evaluation_results = retrieval_evaluator.compute_metrics(
        retrieval_results
    )
    print("Retrieval evaluation done.")
    for key, value in retrieval_evaluation_results.items():
        if key == "detail":
            continue
    
    # Save the retrieval evaluation results
    with open(args.retrieval_eval_output_file, "w") as f:
        json.dump(retrieval_evaluation_results, f, indent=4)
    print(f"Retrieval evaluation results saved to {args.retrieval_eval_output_file}")
