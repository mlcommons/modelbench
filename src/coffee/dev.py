import click
import pathlib
from datetime import datetime, timedelta, timezone

from coffee.static_site_generator import StaticSiteGenerator

from coffee.newhelm_runner import NewhelmSut
from coffee.benchmark import (
    GeneralChatBotBenchmarkDefinition,
    BiasHazardDefinition,
    HazardScore,
    BenchmarkScore,
    ToxicityHazardDefinition,
)

start_time = datetime.now(timezone.utc)
end_time = start_time + timedelta(minutes=2)


def benchmark_scores():
    bd = GeneralChatBotBenchmarkDefinition()
    bs = [
        BenchmarkScore(
            bd,
            NewhelmSut.GPT2,
            [
                HazardScore(BiasHazardDefinition(), 0.5, start_time=start_time, end_time=end_time),
                HazardScore(ToxicityHazardDefinition(), 0.8, start_time=start_time, end_time=end_time),
            ],
            start_time=start_time,
            end_time=end_time,
        ),
        BenchmarkScore(
            bd,
            NewhelmSut.LLAMA_2_7B,
            [
                HazardScore(BiasHazardDefinition(), 0.5, start_time=start_time, end_time=end_time),
                HazardScore(ToxicityHazardDefinition(), 0.8, start_time=start_time, end_time=end_time),
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
