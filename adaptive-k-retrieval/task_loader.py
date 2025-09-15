from datasets import Dataset, load_dataset, load_from_disk
from typing import Literal, List, Optional
import gc

# local imports
from load_context import load_context


class TaskLoader():
    """A class for loading a task from a file."""
    def __init__(self,
                 debug: bool = False):
        self.debug = debug

    def read_file(self,
                  file_path: str) -> Dataset:
        """Read a file."""
        if file_path.endswith("json"):
            data: Dataset = load_dataset("json",
                                         data_files=file_path,
                                         field="data")["train"]
        elif file_path.endswith("jsonl"):
            data: Dataset = load_dataset("json",
                                         data_files=file_path)["train"]
        else:
            raise ValueError(
                "The specified dataset file extension is not supported."
            )
        return data

    def preprocess_qa(self,
                      batch: dict,
                      idx: int) -> dict:
        """Preprocess a batch of Dataset.
        Convert the batch to contain:
        - 'idx' (index)
        - 'query' ('question')
        - 'label' ('answer')
        - 'context'
        - 'positive_context' (correct claim)
        - 'relevant_context'
        
        Args:
            batch (dict): A batch of Dataset.
            idx (int): The index of the batch.
        Returns:
            batch (dict): The preprocessed batch.
        """
        batch["idx"] = idx
        batch["query"] = batch["question"]
        batch["label"] = batch["answers"]
        batch["context"] = batch["ctxs"]
        batch["positive_context"] = batch["positive_ctxs"]
        batch["relevant_context"] = batch["hard_negative_ctxs"]
        return batch

    def load_qa(self,
                file_path: str,
                num_samples: Optional[int],
                drop_duplicates: bool) -> Dataset:
        """Load a QA task.

        Args:
            file_path (str): The path to the dataset file.
            num_samples (Optional[int]): The number of samples to load.
                If None, load all samples.

        Returns:
            Dataset: The loaded dataset.
                A Dataset sample (dict) consists of 'query', 'context', 'answer', 'positive_context', and 'relevant_context'.
                If self.debug, return a dataset with only one sample.
        """
        data: Dataset = self.read_file(file_path)
        data: Dataset = data.map(
            self.preprocess_qa,
            remove_columns=data.column_names,
            with_indices=True # This is necessary for adding the indices to the dataset
            )

        if drop_duplicates:
            print("Dropping duplicates...")
            df = data.to_pandas()
            df = df.drop_duplicates(subset=["query"])
            data = Dataset.from_pandas(df)

            # garbage collection
            del df
            gc.collect()

        if self.debug:
            return data.take(10)
        elif num_samples:
            return data.take(num_samples)
        else:
            return data

    def load_holobench(self,
                       split: Literal[
                           "wine_1",
                           "college_2",
                           "flight_4",
                           "store_1",
                           "soccer_1",
                           "all"
                       ],
                       max_context_size: int,
                       info_amount: int,
                       merge_strategy: Literal[
                           "uniform",
                           "begin",
                           "end",
                           "middle",
                           "bimodal"
                       ],
                       num_samples: Optional[int],
                       preprocessed_data: Optional[str]) -> Dataset:
        """Load HoloBench.

        Structure of the HoloBench dataset:
        DatasetDict({
            wine_1: Dataset({
                features: ['question', 'query', 'db_id', 'difficulty', 'SQL_COMPONENTS_1', 'SQL_COMPONENTS_2', 'Other Complexity Factors', 'query_type', 'Aggregation', 'Max/Min', 'Join', 'Comparison', 'Ranking'],
                num_rows: 18
            })
            college_2: Dataset({...})
            flight_4: Dataset({...})
            store_1: Dataset({...})
            soccer_1: Dataset({...})
        })

        Args:
            split (str): The split of the dataset to load.
            max_context_size (int): The maximum number of context sentences to load.
            info_amount (int): The amount of information to load.
            merge_strategy (str): The strategy to merge the context sentences.
                The strategy to merge the context sentences

        Returns:
            Dataset: The loaded dataset.
                A Dataset sample (dict) consists of 'query', 'context', 'answer', 'positive_context', and 'relevant_context'.
                If self.debug, return a dataset with only one sample.
        """
        if preprocessed_data is not None:
            data = load_from_disk(preprocessed_data)
        
        else:
            # for questions
            queries = load_dataset("megagonlabs/holobench",
                                name="queries")

            if split == "all":
                all_splits = ["wine_1", "college_2",
                            "flight_4", "store_1", "soccer_1"]
            else:
                all_splits = [split]

            datasets: List[dict] = []

            idx = 0
            for sp in all_splits:
                # for generating contexts
                db = load_dataset("megagonlabs/holobench",
                                name=sp)
                for ins in queries[sp]:
                    context, gold_answer, relevant_indices = load_context(
                        ins["query"],
                        db,
                        max_context_size=max_context_size,
                        info_amount=info_amount,
                        merge_strategy=merge_strategy
                    )
                    # context: List[str]
                    # gold_answer: pd.DataFrame
                    # relevant_indices: List[int]

                    if isinstance(context, str):
                        context = context.split("\n")

                    # make context into List[Dict[str, str]]
                    context = [{"psg_id": i,
                                "title": sp,
                                "text": c} for i, c in enumerate(context)]
                    
                    relevant_context = [
                        c for c in context if c["psg_id"] in relevant_indices
                    ]

                    sample = {
                        "idx": idx,
                        "query": ins["question"],
                        "label": gold_answer.to_json(orient="records"),
                        "context": context,
                        "positive_context": relevant_context,
                        "relevant_context": None # there's no hard-negative context
                        }
                    datasets.append(sample)
                    idx += 1

                    if self.debug:
                        # return with just one sample
                        return Dataset.from_list(datasets)

            data = Dataset.from_list(datasets)
        
        if num_samples:
            return data.take(min(num_samples, len(data)))
        return data
