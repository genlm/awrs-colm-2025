import click
import asyncio
from rich import console
from genlm.eval import run_evaluation
from genlm.backend import load_model_by_name

from .tasks import Task, TASK_REGISTRY
from .methods import BaseLM, LCD, SampleRerank, TwistedSMC, AWRSSMC

console = console.Console()

TASKS = list(TASK_REGISTRY.keys())


def common_options(f):
    """Shared CLI options across all methods"""
    f = click.option(
        "--task",
        type=click.Choice(TASKS),
        required=True,
        help="Name of the task to run.",
    )(f)
    f = click.option(
        "--model-name",
        required=True,
        help="Name of the model to use.",
    )(f)
    f = click.option(
        "--result-dir",
        default=None,
        help="Directory to write the inference results.",
    )(f)
    f = click.option(
        "--overwrite-results",
        is_flag=True,
        help="Overwrite existing evaluation results.",
    )(f)
    f = click.option(
        "--overwrite-outputs",
        is_flag=True,
        help="Overwrite existing inference output.",
    )(f)
    f = click.option(
        "--verbosity",
        default=1,
        type=int,
        help="Verbosity level for evaluation. 0 is silent, 1 is verbose.",
    )(f)
    f = click.option(
        "--max-n-instances",
        default=100000,
        type=int,
        help="Maximum number of instances in the dataset to evaluate.",
    )(f)
    f = click.option(
        "--use-chat-format",
        is_flag=True,
        help="Use chat template for the prompt.",
    )(f)

    return f


@click.command()
@common_options
def base_lm(
    task,
    model_name,
    result_dir,
    overwrite_results,
    overwrite_outputs,
    verbosity,
    max_n_instances,
    use_chat_format,
):
    """Run the base language model on a task."""
    console.print(f"[blue]Running base LM on '{task}'[/blue]")
    task = Task.from_name(task)
    model = BaseLM(load_model_by_name(model_name), task, use_chat_format)
    asyncio.run(
        run_evaluation(
            dataset=task.dataset,
            model=model,
            evaluator=task.evaluator,
            n_replicates=1,
            verbosity=verbosity,
            output_dir=result_dir,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            max_instances=max_n_instances,
        )
    )


@click.command()
@common_options
def lcd(
    task,
    model_name,
    result_dir,
    overwrite_results,
    overwrite_outputs,
    verbosity,
    max_n_instances,
    use_chat_format,
):
    """Run locally constrained decoding on a task."""
    console.print(f"[blue]Running LCD on '{task}'[/blue]")
    task = Task.from_name(task)
    asyncio.run(
        run_evaluation(
            dataset=task.dataset,
            model=LCD(load_model_by_name(model_name), task, use_chat_format),
            evaluator=task.evaluator,
            n_replicates=1,
            verbosity=verbosity,
            output_dir=result_dir,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            max_instances=max_n_instances,
        )
    )


@click.command()
@common_options
@click.option(
    "--num-particles", type=int, required=True, help="Number of particles to use"
)
def sample_rerank(
    task,
    model_name,
    result_dir,
    overwrite_results,
    overwrite_outputs,
    verbosity,
    max_n_instances,
    num_particles,
    use_chat_format,
):
    """Run sample rerank on a task."""
    console.print(
        f"[blue]Running sample rerank on '{task}' with {num_particles} particles[/blue]"
    )
    task = Task.from_name(task)
    model = SampleRerank(
        load_model_by_name(model_name), task, num_particles, use_chat_format
    )
    asyncio.run(
        run_evaluation(
            dataset=task.dataset,
            model=model,
            evaluator=task.evaluator,
            n_replicates=1,
            verbosity=verbosity,
            output_dir=result_dir,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            max_instances=max_n_instances,
        )
    )


@click.command()
@common_options
@click.option(
    "--num-particles", type=int, required=True, help="Number of particles to use"
)
@click.option("--ess-threshold", type=float, required=True, help="ESS threshold")
def twisted_smc(
    task,
    model_name,
    result_dir,
    overwrite_results,
    overwrite_outputs,
    verbosity,
    max_n_instances,
    num_particles,
    ess_threshold,
    use_chat_format,
):
    """Run twisted SMC on a task."""
    console.print(
        f"[blue]Running twisted SMC on '{task}' with {num_particles} particles and ESS threshold {ess_threshold}[/blue]"
    )
    task = Task.from_name(task)
    model = TwistedSMC(
        load_model_by_name(model_name),
        task,
        num_particles,
        ess_threshold,
        use_chat_format,
    )
    asyncio.run(
        run_evaluation(
            dataset=task.dataset,
            model=model,
            evaluator=task.evaluator,
            n_replicates=1,
            verbosity=verbosity,
            output_dir=result_dir,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            max_instances=max_n_instances,
        )
    )


@click.command()
@common_options
@click.option(
    "--num-particles", type=int, required=True, help="Number of particles to use"
)
@click.option("--ess-threshold", type=float, required=True, help="ESS threshold")
def awrs_smc(
    task,
    model_name,
    result_dir,
    overwrite_results,
    overwrite_outputs,
    verbosity,
    max_n_instances,
    use_chat_format,
    num_particles,
    ess_threshold,
):
    """Run AWRS SMC on a task."""
    console.print(
        f"[blue]Running AWRS SMC on '{task}' with {num_particles} particles and ESS threshold {ess_threshold}[/blue]"
    )
    task = Task.from_name(task)
    model = AWRSSMC(
        load_model_by_name(model_name),
        task,
        num_particles,
        ess_threshold,
        use_chat_format,
    )
    asyncio.run(
        run_evaluation(
            dataset=task.dataset,
            model=model,
            evaluator=task.evaluator,
            n_replicates=1,
            verbosity=verbosity,
            output_dir=result_dir,
            overwrite_results=overwrite_results,
            overwrite_outputs=overwrite_outputs,
            max_instances=max_n_instances,
        )
    )


@click.group()
def cli():
    pass


cli.add_command(base_lm)
cli.add_command(lcd)
cli.add_command(sample_rerank)
cli.add_command(twisted_smc)
cli.add_command(awrs_smc)

if __name__ == "__main__":
    cli()
