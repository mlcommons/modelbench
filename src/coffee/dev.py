import pathlib
from datetime import datetime, timedelta, timezone

import click
from newhelm.config import load_secrets_from_config

from coffee.benchmark import (
    BenchmarkScore,
    BiasHazardDefinition,
    GeneralChatBotBenchmarkDefinition,
    HazardScore,
    ToxicityHazardDefinition,
    ValueEstimate,
)
from coffee.newhelm_runner import NewhelmSut
from coffee.static_site_generator import StaticSiteGenerator

start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(minutes=2)


def benchmark_scores():

    secrets = load_secrets_from_config()

    bd = GeneralChatBotBenchmarkDefinition(secrets=secrets)
    bs = [
        BenchmarkScore(
            bd,
            NewhelmSut.GPT2,
            [
                HazardScore(hazard_definition=BiasHazardDefinition(), score=ValueEstimate.make([0.5]), test_scores={}),
                HazardScore(
                    hazard_definition=ToxicityHazardDefinition(secrets=secrets),
                    score=ValueEstimate.make([0.8]),
                    test_scores={},
                ),
            ],
            start_time=start_time,
            end_time=end_time,
        ),
        BenchmarkScore(
            bd,
            NewhelmSut.LLAMA_2_7B,
            [
                HazardScore(hazard_definition=BiasHazardDefinition(), score=ValueEstimate.make([0.5]), test_scores={}),
                HazardScore(
                    hazard_definition=ToxicityHazardDefinition(secrets=secrets),
                    score=ValueEstimate.make([0.8]),
                    test_scores={},
                ),
            ],
            start_time=start_time,
            end_time=end_time,
        ),
    ]
    return bs


@click.group()
def cli() -> None:
    pass


@cli.command(help="Generate a simple site for development")
@click.option(
    "--output-dir", "-o", default="./web", type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path)
)
@click.option("--view-embed", "-e", is_flag=True, default=False)
def build(output_dir: pathlib.Path, view_embed: bool) -> None:
    generator = StaticSiteGenerator(view_embed)
    generator.generate(benchmark_scores(), output_dir)


if __name__ == "__main__":
    cli()
