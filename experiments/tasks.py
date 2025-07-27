from dataclasses import dataclass
from typing import Protocol
from transformers import AutoTokenizer

from genlm.control import Potential
from genlm.backend.llm import AsyncVirtualLM
from genlm.eval import Dataset, Evaluator, Instance


class PotentialFactory(Protocol):
    """A factory for creating potentials given a dataset instance."""

    def __call__(self, instance: Instance) -> Potential: ...


class EOSFactory(Protocol):
    """A factory for creating eos tokens given the language model."""

    def __call__(self, llm: AsyncVirtualLM) -> list[bytes]: ...


class PromptFactory(Protocol):
    """A factory for creating prompts given an instance."""

    def __call__(
        self,
        tokenizer: AutoTokenizer,
        instance: Instance,
        use_chat_template: bool = True,
    ) -> str: ...


@dataclass
class Task:
    """A task configuration."""

    name: str
    dataset: Dataset
    evaluator: Evaluator
    max_tokens: int
    potential_factory: PotentialFactory
    eos_token_factory: EOSFactory
    prompt_factory: PromptFactory

    @classmethod
    def from_name(cls, name: str) -> "Task":
        """Load the default task configuration for the given name."""
        if name == "pattern-matching":
            return cls.pattern_matching()
        elif name == "text-to-sql":
            return cls.text_to_sql()
        elif name == "molecular-synthesis":
            return cls.molecular_synthesis()
        elif name == "json":
            return cls.json()
        elif name == "goal-inference":
            return cls.goal_inference()
        else:
            raise ValueError(f"Unknown task: {name}")

    @classmethod
    def pattern_matching(
        cls,
        pattern_csv_path: str = "data/pattern_matching/patterns.csv",
        max_tokens: int = 100,
    ) -> "Task":
        """Load the task configuration for pattern matching."""
        from genlm.eval.domains.pattern_matching import (
            PatternMatchingDataset,
            PatternMatchingEvaluator,
            PatternPotential,
            default_prompt_formatter,
        )

        def eos_token_factory(llm) -> list[bytes]:
            eos_tokens = []

            # Special tokens
            for token_id in llm.model.tokenizer.get_added_vocab().values():
                eos_tokens.append(llm.token_maps.decode[token_id])

            # Newlines
            for token in llm.vocab:
                if b"\n" in token:
                    eos_tokens.append(token)

            eos_tokens.append(llm.token_maps.decode[llm.model.tokenizer.eos_token_id])

            return list(set(eos_tokens))

        return cls(
            name="pattern-matching",
            dataset=PatternMatchingDataset.from_csv(
                pattern_csv_path, pattern_column="regex"
            ),
            evaluator=PatternMatchingEvaluator(),
            potential_factory=lambda instance: PatternPotential(instance.pattern),
            eos_token_factory=eos_token_factory,
            prompt_factory=default_prompt_formatter,
            max_tokens=max_tokens,
        )

    @classmethod
    def text_to_sql(cls, spider_data_dir: str, spider_grammars: str) -> "Task":
        """Load the task configuration for text-to-SQL."""
        from genlm.control import BoolCFG
        from genlm.eval.domains.spider import (
            SpiderDataset,
            SpiderEvaluator,
            default_prompt_formatter,
        )

        return cls(
            name="text-to-sql",
            dataset=SpiderDataset.from_spider_dir(
                spider_data_dir, grammar_json_path=spider_grammars
            ),
            evaluator=SpiderEvaluator(spider_data_dir),
            potential_factory=lambda instance: BoolCFG.from_lark(instance.lark_grammar),
            eos_token_factory=lambda vocab: [b"\n", b"\n\n"],
            prompt_factory=default_prompt_formatter,
        )

    @classmethod
    def molecular_synthesis(cls, molecular_synthesis_csv_path: str) -> "Task":
        """Load the task configuration for molecular synthesis."""
        from genlm.eval.domains.molecular_synthesis import (
            MolecularSynthesisDataset,
            MolecularSynthesisEvaluator,
            default_prompt_formatter,
            PartialSMILES,
        )

        return cls(
            name="molecular-synthesis",
            dataset=MolecularSynthesisDataset.from_csv(molecular_synthesis_csv_path),
            evaluator=MolecularSynthesisEvaluator(),
            potential_factory=lambda instance: PartialSMILES(),
            eos_token_factory=lambda vocab: [b"\n", b"\n\n"],
            prompt_factory=default_prompt_formatter,
        )

    @classmethod
    def json(cls):
        raise NotImplementedError("JSON is not implemented yet")

    @classmethod
    def goal_inference(cls):
        raise NotImplementedError("Goal inference is not implemented yet")
