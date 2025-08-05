import pytest

from experiments.methods import BaseLM, LCD, SampleRerank, TwistedSMC, AWRSSMC
from experiments.tasks import (
    PatternMatching,
    MolecularSynthesis,
    TextToSQL,
    JSON,
    DATA_DIR,
)


TASKS = {}
for task, use_chat_format in [
    (PatternMatching(), True),
    (
        MolecularSynthesis(
            smiles_path=DATA_DIR / "molecular_synthesis" / "GDB17_sample.txt",
        ),
        False,
    ),
    (
        TextToSQL(
            spider_data_dir=DATA_DIR / "spider" / "spider_sample",
            few_shot_example_ids=[0, 1],
        ),
        True,
    ),
    (JSON(), True),
]:
    TASKS[task.__class__.__name__] = (task, next(iter(task.dataset)), use_chat_format)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_base_lm(llm, task_name):
    task, instance, use_chat_format = TASKS[task_name]
    method = BaseLM(llm, task, use_chat_format=use_chat_format)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_lcd(llm, task_name):
    task, instance, use_chat_format = TASKS[task_name]
    method = LCD(llm, task, use_chat_format=use_chat_format)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_sample_rerank(llm, task_name):
    task, instance, use_chat_format = TASKS[task_name]
    method = SampleRerank(llm, task, n_particles=2, use_chat_format=use_chat_format)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_twisted_smc(llm, task_name):
    task, instance, use_chat_format = TASKS[task_name]
    method = TwistedSMC(
        llm, task, n_particles=2, ess_threshold=0.5, use_chat_format=use_chat_format
    )
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_awrs_smc(llm, task_name):
    task, instance, use_chat_format = TASKS[task_name]
    method = AWRSSMC(
        llm, task, n_particles=2, ess_threshold=0.5, use_chat_format=use_chat_format
    )
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0
