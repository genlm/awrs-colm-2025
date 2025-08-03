from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from functools import lru_cache
from genlm.control import Potential, BoolCFG
from genlm.backend.llm import AsyncVirtualLM
from genlm.eval import Dataset, Evaluator, Instance

from genlm.eval.domains import (
    molecular_synthesis,
    pattern_matching,
    spider,
)

DATA_DIR = Path(__file__).parent.parent / "data"


class Task(ABC):
    """A task configuration."""

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        evaluator: Evaluator,
        max_tokens: int,
    ):
        self.name = name
        self.dataset = dataset
        self.evaluator = evaluator
        self.max_tokens = max_tokens

    @abstractmethod
    def make_condition(self, instance: Instance) -> Potential:
        pass

    @abstractmethod
    def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
        pass

    @abstractmethod
    def get_prompt(self, tokenizer: AutoTokenizer, instance: Instance) -> list[int]:
        pass


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
            "pattern-matching",
            pattern_matching.PatternMatchingDataset.from_csv(
                pattern_csv_path, pattern_column="regex"
            ),
            pattern_matching.PatternMatchingEvaluator(),
            max_tokens,
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

    def get_prompt(self, tokenizer: AutoTokenizer, instance: Instance) -> list[int]:
        return pattern_matching.default_prompt_formatter(tokenizer, instance)


class TextToSQL(Task):
    """A task configuration for text-to-SQL."""

    def __init__(
        self,
        spider_data_dir: Optional[str] = None,
        spider_grammars: Optional[str] = None,
        few_shot_example_ids: Optional[list[int]] = None,
        max_tokens: int = 100,
    ):
        spider_data_dir = spider_data_dir or DATA_DIR / "spider"
        spider_grammars = spider_grammars or DATA_DIR / "spider" / "grammars.json"

        super().__init__(
            name="text-to-sql",
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

    def get_prompt(self, tokenizer: AutoTokenizer, instance: Instance) -> list[int]:
        from genlm.eval.domains.spider import default_prompt_formatter

        return default_prompt_formatter(tokenizer, instance)


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
            name="molecular-synthesis",
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

    def get_prompt(self, tokenizer: AutoTokenizer, instance: Instance) -> list[int]:
        return molecular_synthesis.default_prompt_formatter(tokenizer, instance)


# class JSON(Task):
#     """A task configuration for JSON."""

#     def __init__(self, json_data_path: str, max_tokens: int = 450):
#         super().__init__(
#             name="json",
#             dataset=json.JSONDataset.from_csv(json_data_path),
#             evaluator=json.JSONEvaluator(),
#             max_tokens=max_tokens,
#         )

#     def make_potential(self, instance: Instance) -> Potential:
#         return JsonSchema(instance.schema)

#     def get_eos_tokens(self, llm: AsyncVirtualLM) -> list[bytes]:
#         return [llm.tokenizer.eos_token_id]

#     def get_prompt(self, tokenizer: AutoTokenizer, instance: Instance) -> list[int]:
#         return json.default_prompt_formatter(tokenizer, instance)
