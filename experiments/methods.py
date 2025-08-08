"""Sampling methods."""

from functools import cached_property
from abc import ABC, abstractmethod
import time

from genlm.backend.llm import AsyncVirtualLM
from genlm.eval import Instance, ModelOutput, ModelResponse
from genlm.control import direct_token_sampler, PromptedLLM, Potential
from genlm.control.sampler.sequence import Sequences

from .sampler import AWRS
from .tasks import Task


class Method(ABC):
    def __init__(
        self,
        llm: AsyncVirtualLM,
        task: Task,
        use_chat_format: bool = True,
        seed: int = None,
    ):
        self.llm = llm
        self.task = task
        self.use_chat_format = use_chat_format
        self.seed = seed

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    @cached_property
    def eos_tokens(self) -> list[bytes]:
        return self.task.get_eos_tokens(self.llm)

    @property
    def max_tokens(self) -> int:
        return self.task.max_tokens

    @staticmethod
    def postprocess(sequences: Sequences, runtime: float) -> ModelOutput:
        return ModelOutput(
            responses=[
                ModelResponse(response=sequence, weight=prob)
                for sequence, prob in sequences.decoded_posterior.items()
            ],
            runtime_seconds=runtime,
        )

    def make_llm_potential(self, instance: Instance) -> PromptedLLM:
        return PromptedLLM(
            self.llm,
            prompt_ids=self.task.get_prompt(
                tokenizer=self.tokenizer,
                instance=instance,
                use_chat_format=self.use_chat_format,
            ),
            eos_tokens=self.eos_tokens,
        )

    def make_potentials(self, instance: Instance) -> tuple[Potential, Potential]:
        llm_potential = self.make_llm_potential(instance)
        condition = self.task.make_condition(instance).coerce(llm_potential, f=b"".join)
        return llm_potential, condition

    @abstractmethod
    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        pass


class BaseLM(Method):
    """Sample directly from the language model."""

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        llm = self.make_llm_potential(instance)

        sampler = direct_token_sampler(llm)

        start = time.time()
        outputs = await sampler.smc(
            n_particles=1,
            ess_threshold=0,
            max_tokens=self.max_tokens,
        )
        runtime = time.time() - start

        return self.postprocess(outputs, runtime)


class LCD(Method):
    """Sample using locally constrained decoding."""

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        llm_potential, condition = self.make_potentials(instance)

        seed = self.seed + replicate if self.seed is not None else None

        sampler = AWRS(llm_potential, condition, proper_weights=False, seed=seed)

        start = time.time()
        outputs = await sampler.smc(
            n_particles=1,
            ess_threshold=0,
            max_tokens=self.max_tokens,
        )
        runtime = time.time() - start

        return self.postprocess(outputs, runtime)


class SampleRerank(Method):
    """Importance sampling with the base model as a proposal."""

    def __init__(
        self,
        llm: AsyncVirtualLM,
        task: Task,
        n_particles: int,
        use_chat_format: bool = True,
        seed: int = None,
    ):
        super().__init__(llm, task, use_chat_format, seed)
        self.n_particles = n_particles

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        llm_potential, condition = self.make_potentials(instance)

        sampler = direct_token_sampler(llm_potential)

        start = time.time()
        outputs = await sampler.smc(
            n_particles=self.n_particles,
            ess_threshold=0,
            max_tokens=self.max_tokens,
            critic=condition,
        )
        runtime = time.time() - start

        return self.postprocess(outputs, runtime)


class TwistedSMC(Method):
    """SMC with the base model as a proposal and the condition as twists."""

    def __init__(
        self,
        llm: AsyncVirtualLM,
        task: Task,
        n_particles: int,
        ess_threshold: float,
        use_chat_format: bool = True,
        seed: int = None,
    ):
        super().__init__(llm, task, use_chat_format, seed)
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        llm_potential, condition = self.make_potentials(instance)

        sampler = direct_token_sampler(llm_potential)

        start = time.time()
        outputs = await sampler.smc(
            n_particles=self.n_particles,
            ess_threshold=self.ess_threshold,
            max_tokens=self.max_tokens,
            critic=condition,
        )
        runtime = time.time() - start

        return self.postprocess(outputs, runtime)


class AWRSSMC(Method):
    """SMC with AWRS as a proposal."""

    def __init__(
        self,
        llm: AsyncVirtualLM,
        task: Task,
        n_particles: int,
        ess_threshold: float,
        use_chat_format: bool = True,
        seed: int = None,
    ):
        super().__init__(llm, task, use_chat_format, seed)
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold

    async def __call__(
        self, instance: Instance, output_dir: str, replicate: int
    ) -> ModelOutput:
        llm_potential, condition = self.make_potentials(instance)

        seed = self.seed + replicate if self.seed is not None else None

        sampler = AWRS(llm_potential, condition, proper_weights=True, seed=seed)

        start = time.time()
        outputs = await sampler.smc(
            n_particles=self.n_particles,
            ess_threshold=self.ess_threshold,
            max_tokens=self.max_tokens,
        )
        runtime = time.time() - start

        return self.postprocess(outputs, runtime)
