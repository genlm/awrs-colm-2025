import pytest
from unittest.mock import patch
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


def test_base_lm(llm, runner):
    with patch(
        "experiments.__main__.load_model_by_name",
        return_value=llm,
    ):
        from experiments.__main__ import cli

        result = runner.invoke(
            cli,
            [
                "base-lm",
                "--task",
                "pattern-matching",
                "--model-name",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--use-chat-format",
                "--max-n-instances",
                "3",
            ],
        )

        assert result.exit_code == 0


def test_lcd(llm, runner):
    with patch(
        "experiments.__main__.load_model_by_name",
        return_value=llm,
    ):
        from experiments.__main__ import cli

        result = runner.invoke(
            cli,
            [
                "lcd",
                "--task",
                "pattern-matching",
                "--model-name",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--use-chat-format",
                "--max-n-instances",
                "3",
            ],
        )

        assert result.exit_code == 0


def test_sample_rerank(llm, runner):
    with patch(
        "experiments.__main__.load_model_by_name",
        return_value=llm,
    ):
        from experiments.__main__ import cli

        result = runner.invoke(
            cli,
            [
                "sample-rerank",
                "--task",
                "pattern-matching",
                "--model-name",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--use-chat-format",
                "--max-n-instances",
                "3",
                "--num-particles",
                "2",
            ],
        )

        assert result.exit_code == 0


def test_twisted_smc(llm, runner):
    with patch(
        "experiments.__main__.load_model_by_name",
        return_value=llm,
    ):
        from experiments.__main__ import cli

        result = runner.invoke(
            cli,
            [
                "twisted-smc",
                "--task",
                "pattern-matching",
                "--model-name",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--use-chat-format",
                "--max-n-instances",
                "3",
                "--num-particles",
                "2",
                "--ess-threshold",
                "0.5",
            ],
        )

        assert result.exit_code == 0


def test_awrs_smc(llm, runner):
    with patch(
        "experiments.__main__.load_model_by_name",
        return_value=llm,
    ):
        from experiments.__main__ import cli

        result = runner.invoke(
            cli,
            [
                "awrs-smc",
                "--task",
                "pattern-matching",
                "--model-name",
                "meta-llama/Llama-3.2-1B-Instruct",
                "--use-chat-format",
                "--max-n-instances",
                "3",
                "--num-particles",
                "2",
                "--ess-threshold",
                "0.5",
            ],
        )

        assert result.exit_code == 0
