from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from functools import lru_cache
from genlm.control import Potential, BoolCFG, JsonSchema
from genlm.backend.llm import AsyncVirtualLM
from genlm.eval import Dataset, Evaluator, Instance

from genlm.eval.domains import (
    molecular_synthesis,
    pattern_matching,
    spider,
    json_schema,
)

DATA_DIR = Path(__file__).parent.parent / "data"


TASK_REGISTRY = {}


def register_task(name: str):
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls

    return decorator


class Task(ABC):
    """A task configuration."""

    def __init__(
        self,
        dataset: Dataset,
        evaluator: Evaluator,
        max_tokens: int,
    ):
        self.dataset = dataset
        self.evaluator = evaluator
        self.max_tokens = max_tokens

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> "Task":
        if name not in TASK_REGISTRY:
            raise ValueError(f"Task {name} not found in TASK_REGISTRY")
        return TASK_REGISTRY[name](*args, **kwargs)

    @abstractmethod
    def make_condition(self, instance: Instance) -> Potential:
        pass

    @abstractmethod
    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        pass

    @abstractmethod
    def get_prompt(
        self, tokenizer: AutoTokenizer, instance: Instance, use_chat_format: bool = True
    ) -> list[int]:
        pass


@register_task("pattern-matching")
class PatternMatching(Task):
    """A task configuration for pattern matching."""

    def __init__(
        self,
        pattern_csv_path: Optional[str] = None,
        max_tokens: int = 100,
    ):
        pattern_csv_path = (
            pattern_csv_path or DATA_DIR / "pattern_matching" / "patterns.csv"
        )
        super().__init__(
            dataset=pattern_matching.PatternMatchingDataset.from_csv(
                pattern_csv_path, pattern_column="regex"
            ),
            evaluator=pattern_matching.PatternMatchingEvaluator(),
            max_tokens=max_tokens,
        )

    def make_condition(self, instance: Instance) -> Potential:
        return pattern_matching.PatternPotential(instance.pattern)

    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        eos_tokens = []

        # Special tokens
        for token_id in llm.tokenizer.get_added_vocab().values():
            eos_tokens.append(llm.byte_vocab[token_id])

        # Newlines
        for token in llm.byte_vocab:
            if b"\n" in token:
                eos_tokens.append(token)

        eos_tokens.append(llm.byte_vocab[llm.tokenizer.eos_token_id])

        return list(set(eos_tokens))

    def get_prompt(
        self, tokenizer: AutoTokenizer, instance: Instance, use_chat_format: bool = True
    ) -> list[int]:
        return pattern_matching.default_prompt_formatter(
            tokenizer, instance, use_chat_format
        )


@register_task("text-to-sql")
class TextToSQL(Task):
    """A task configuration for text-to-SQL."""

    def __init__(
        self,
        spider_data_dir: Optional[str] = None,
        spider_grammars: Optional[str] = None,
        few_shot_example_ids: Optional[list[int]] = None,
        max_tokens: int = 100,
    ):
        spider_data_dir = spider_data_dir or (DATA_DIR / "spider" / "spider_data")
        spider_grammars = spider_grammars or (DATA_DIR / "spider" / "grammars.json")

        super().__init__(
            dataset=spider.SpiderDataset.from_spider_dir(
                spider_data_dir,
                grammar_json_path=spider_grammars,
                few_shot_example_ids=few_shot_example_ids,
            ),
            evaluator=spider.SpiderEvaluator(spider_data_dir),
            max_tokens=max_tokens,
        )

    @lru_cache(maxsize=2)
    def _get_cfg(self, lark_grammar: str) -> BoolCFG:
        return BoolCFG.from_lark(lark_grammar)

    def make_condition(self, instance: Instance) -> Potential:
        return self._get_cfg(instance.lark_grammar)

    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        return [llm.byte_vocab[llm.tokenizer.eos_token_id]]

    def get_prompt(
        self, tokenizer: AutoTokenizer, instance: Instance, use_chat_format: bool = True
    ) -> list[int]:
        return spider.default_prompt_formatter(tokenizer, instance, use_chat_format)


@register_task("molecular-synthesis")
class MolecularSynthesis(Task):
    """A task configuration for molecular synthesis."""

    def __init__(
        self,
        smiles_path: Optional[str] = None,
        max_tokens: int = 100,
    ):
        smiles_path = (
            smiles_path or DATA_DIR / "molecular_synthesis" / "GDB17.50000000.smi"
        )

        super().__init__(
            dataset=molecular_synthesis.MolecularSynthesisDataset.from_smiles(
                smiles_path
            ),
            evaluator=molecular_synthesis.MolecularSynthesisEvaluator(),
            max_tokens=max_tokens,
        )

    def make_condition(self, instance: Instance) -> Potential:
        return molecular_synthesis.PartialSMILES()

    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        return [t for t in llm.byte_vocab if b"\n" in t]

    def get_prompt(
        self,
        tokenizer: AutoTokenizer,
        instance: Instance,
        use_chat_format: bool = False,
    ) -> list[int]:
        return molecular_synthesis.default_prompt_formatter(
            tokenizer, instance, use_chat_format
        )


@register_task("json")
class JSON(Task):
    """A task configuration for generating JSON conforming to a schema."""

    def __init__(
        self,
        max_tokens: int = 450,
        tasks: Optional[list[str]] = None,
        split: str = "val",
    ):
        tasks = tasks or ["Github_trivial", "Github_easy", "Github_medium"]
        super().__init__(
            dataset=json_schema.JSONSchemaBenchDataset.from_tasks(tasks, split),
            evaluator=json_schema.JSONSchemaBenchEvaluator(),
            max_tokens=max_tokens,
        )

    def make_condition(self, instance: Instance) -> Potential:
        return JsonSchema(instance.json_schema)

    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        return [llm.byte_vocab[llm.tokenizer.eos_token_id]]

    def get_prompt(
        self, tokenizer: AutoTokenizer, instance: Instance, use_chat_format: bool = True
    ) -> list[int]:
        return json_schema.default_prompt_formatter(
            tokenizer, instance, use_chat_format
        )
