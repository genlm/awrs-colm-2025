import pytest

from experiments.methods import BaseLM, LCD, SampleRerank, TwistedSMC, AWRSSMC
from experiments.tasks import PatternMatching, MolecularSynthesis, TextToSQL, DATA_DIR


TASKS = {}
for task in [
    PatternMatching(),
    MolecularSynthesis(
        smiles_path=DATA_DIR / "molecular_synthesis" / "GDB17_sample.txt",
    ),
    TextToSQL(
        spider_data_dir=DATA_DIR / "spider" / "spider_sample",
        few_shot_example_ids=[0, 1],
    ),
]:
    TASKS[task.__class__.__name__] = (task, next(iter(task.dataset)))


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_base_lm(llm, task_name):
    task, instance = TASKS[task_name]
    method = BaseLM(llm, task)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_lcd(llm, task_name):
    task, instance = TASKS[task_name]
    method = LCD(llm, task)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_sample_rerank(llm, task_name):
    task, instance = TASKS[task_name]
    method = SampleRerank(llm, task, n_particles=2)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_twisted_smc(llm, task_name):
    task, instance = TASKS[task_name]
    method = TwistedSMC(llm, task, n_particles=2, ess_threshold=0.5)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("task_name", TASKS.keys())
async def test_awrs_smc(llm, task_name):
    task, instance = TASKS[task_name]
    method = AWRSSMC(llm, task, n_particles=2, ess_threshold=0.5)
    result = await method(instance)
    assert result is not None
    assert result.runtime_seconds > 0
