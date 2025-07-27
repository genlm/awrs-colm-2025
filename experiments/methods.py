"""Sampling methods."""

from genlm.backend.llm import AsyncVirtualLM
from genlm.eval import Instance, ModelOutput, ModelResponse
from genlm.control import direct_token_sampler, PromptedLLM
from genlm.control.sampler.sequence import Sequences

from .sampler import AWRS
from .tasks import PromptFactory, EOSFactory, PotentialFactory


class Method:
    def __init__(
        self,
        llm: AsyncVirtualLM,
        prompt_factory: PromptFactory,
        eos_token_factory: EOSFactory,
        max_tokens: int,
    ):
        self.llm = llm
        self.tokenizer = self.llm.tokenizer
        self.prompt_factory = prompt_factory
        self.max_tokens = max_tokens
        self.eos_tokens = eos_token_factory(self.llm.byte_vocab)

    @staticmethod
    def postprocess(sequences: Sequences) -> ModelOutput:
        return ModelOutput(
            responses=[
                ModelResponse(response=sequence, weight=prob)
                for sequence, prob in sequences.decoded_posterior.items()
            ],
        )

    def make_llm_potential(self, instance: Instance) -> PromptedLLM:
        return PromptedLLM(
            self.llm,
            prompt_ids=self.prompt_factory(self.tokenizer, instance),
            eos_tokens=self.eos_tokens,
        )

    async def __call__(self, instance: Instance, *args, **kwargs) -> ModelOutput:
        pass


class BaseLM(Method):
    async def __call__(self, instance: Instance) -> ModelOutput:
        return self.postprocess(
            await direct_token_sampler(self.llm).smc(
                n_particles=1,
                ess_threshold=0,
                max_tokens=self.max_tokens,
            )
        )


class LCD(Method):
    def __init__(
        self,
        llm: AsyncVirtualLM,
        prompt_factory: PromptFactory,
        eos_token_factory: EOSFactory,
        condition_factory: PotentialFactory,
        max_tokens: int = 1000,
    ):
        super().__init__(llm, prompt_factory, eos_token_factory, max_tokens)
        self.condition_factory = condition_factory

    async def __call__(self, instance: Instance, *args, **kwargs) -> ModelOutput:
        llm_potential = self.make_llm_potential(instance)
        condition = self.condition_factory(instance).coerce(llm_potential, f="".join)
        return self.postprocess(
            await AWRS(llm_potential, condition, proper_weights=False).smc(
                n_particles=1,
                ess_threshold=0,
                max_tokens=self.max_tokens,
            )
        )


class SampleRerank(Method):
    def __init__(
        self,
        llm: AsyncVirtualLM,
        n_particles: int,
        prompt_factory: PromptFactory,
        eos_token_factory: EOSFactory,
        condition_factory: PotentialFactory,
        max_tokens: int = 1000,
    ):
        super().__init__(llm, prompt_factory, eos_token_factory, max_tokens)
        self.n_particles = n_particles
        self.condition_factory = condition_factory

    async def __call__(self, instance: Instance, *args, **kwargs) -> ModelOutput:
        llm_potential = self.make_llm_potential(instance)
        condition = self.condition_factory(instance).coerce(llm_potential, f="".join)
        return self.postprocess(
            await direct_token_sampler(llm_potential).smc(
                n_particles=self.n_particles,
                ess_threshold=0,
                max_tokens=self.max_tokens,
                critic=condition,
            )
        )


class TwistedSMC(Method):
    def __init__(
        self,
        llm: AsyncVirtualLM,
        n_particles: int,
        ess_threshold: float,
        max_tokens: int = 1000,
    ):
        super().__init__(llm, max_tokens)
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold

    async def __call__(self, instance: Instance, *args, **kwargs) -> ModelOutput:
        llm_potential = self.make_llm_potential(instance)
        condition = self.condition_factory(instance).coerce(llm_potential, f="".join)
        return self.postprocess(
            await direct_token_sampler(self.llm).smc(
                n_particles=self.n_particles,
                ess_threshold=self.ess_threshold,
                max_tokens=self.max_tokens,
                critic=condition,
            )
        )


class AWRSSMC(Method):
    def __init__(
        self,
        llm: AsyncVirtualLM,
        n_particles: int,
        ess_threshold: float,
        max_tokens: int = 1000,
    ):
        super().__init__(llm, max_tokens)
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold

    async def __call__(self, instance: Instance, *args, **kwargs) -> ModelOutput:
        llm_potential = self.make_llm_potential(instance)
        condition = self.condition_factory(instance).coerce(llm_potential, f="".join)
        return self.postprocess(
            await AWRS(llm_potential, condition, proper_weights=True).smc(
                n_particles=self.n_particles,
                ess_threshold=self.ess_threshold,
                max_tokens=self.max_tokens,
            )
        )
