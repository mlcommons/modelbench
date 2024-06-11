import pathlib
import shutil
from collections import defaultdict
from functools import singledispatchmethod
from typing import List, Mapping

import casefy
import tomli
from jinja2 import Environment, PackageLoader, select_autoescape
from modelgauge.base_test import BaseTest

from modelbench.benchmarks import BenchmarkDefinition, BenchmarkRun
from modelbench.hazards import HazardDefinition, HazardRun
from modelbench.modelgauge_runner import SutDescription
from modelbench.scoring import NumericGradeMixin
from modelbench.utilities import group_by_key


# TODO: there exist some highly unlikely edge cases where bars may overlap or exceed their bounds as shown by the tests
# class HazardScorePositions(NumericGradeMixin):
#     def __init__(self, min_bar_width: float, lowest_bar_percent: float):
#         self.min_bar_width = min_bar_width
#         self.lowest_bar_percent = lowest_bar_percent
#
#     def __call__(self, hazard_score: HazardScore) -> dict:
#         return {
#             "grade_bands": self._grade_bands(hazard_score),
#             "point_position": self._point_position(hazard_score, hazard_score.score.estimate),
#             "error_bar": self._error_bar(hazard_score),
#         }
#
#     def _grade_bands(
#         self,
#         hazard_score: HazardScore,
#     ) -> list[tuple[int, int]]:
#         new_grades = [hazard_score.grade_points()[0], hazard_score.grade_points()[1] * self.lowest_bar_percent]
#         for i, grade in enumerate(hazard_score.grade_points()[2:-1]):
#             new_grades.append(
#                 (min(1 - ((3 - i) * self.min_bar_width), 1 - (1 - grade) * (1 / self.lowest_bar_percent)))
#             )
#         new_grades.append(hazard_score.grade_points()[-1])
#
#         bands = [(low * 100, high * 100) for low, high in zip(new_grades, new_grades[1:])]
#         return bands
#
#     def _point_position(self, hazard_score: HazardScore, num) -> float:
#         band_range = self._grade_bands(hazard_score)[self._numeric_grade(hazard_score, num) - 1]
#         grade_range = hazard_score.grade_points()[
#             self._numeric_grade(hazard_score, num) - 1 : self._numeric_grade(hazard_score, num) + 1
#         ]
#         perc = (num - grade_range[0]) / (grade_range[1] - grade_range[0])
#         position = perc * (band_range[1] - band_range[0]) + band_range[0]
#
#         # nudge the final location so it doesn't obscure the edges of a range band
#         if position - band_range[0] < 1.5:
#             return band_range[0] + 1.5
#         elif band_range[1] < 1.5:
#             return band_range[1] - 1.5
#         else:
#             return position
#
#     def _error_bar(self, hazard_score: HazardScore) -> dict:
#         lower = self._point_position(hazard_score, hazard_score.score.lower)
#         estimate = self._point_position(hazard_score, hazard_score.score.estimate)
#         upper = self._point_position(hazard_score, hazard_score.score.upper)
#         return {"start": lower, "width": upper - lower}


class StaticContent(dict):

    def __init__(self, path=pathlib.Path(__file__).parent / "templates" / "content"):
        super().__init__()
        for file in (path).rglob("*.toml"):
            with open(file, "rb") as f:
                try:
                    data = tomli.load(f)
                except tomli.TOMLDecodeError as e:
                    raise ValueError(f"failure reading {file}") from e
                duplicate_keys = set(self.keys()) & set(data.keys())
                if duplicate_keys:
                    raise Exception(f"Duplicate tables found in content files: {duplicate_keys}")
                self.update(data)


class StaticSiteGenerator:
    def __init__(self, view_embed: bool = False) -> None:
        """Initialize the StaticSiteGenerator class for local file or website partial

        Args:
            view_embed (bool): Whether to generate local file or embedded view. Defaults to False.
        """
        self.view_embed = view_embed
        self.env = Environment(loader=PackageLoader("modelbench"), autoescape=select_autoescape())
        self.env.globals["hsp"] = HazardScorePositions(min_bar_width=0.04, lowest_bar_percent=0.2)
        self.env.globals["root_path"] = self.root_path
        self.env.globals["benchmarks_path"] = self.benchmarks_path
        self.env.globals["benchmark_path"] = self.benchmark_path
        self.env.globals["test_report_path"] = self.test_report_path
        self.env.globals["content"] = self.content
        self._content = StaticContent()

    @singledispatchmethod
    def content(self, item, key: str):
        pass

    @content.register
    def content_benchmark(self, item: BenchmarkDefinition, key: str):
        content = self._content[item.path_name()]
        try:
            return content[key]
        except KeyError as e:
            raise KeyError(f"{key} not found in {item.path_name()} among {sorted(content.keys())}") from e

    @content.register
    def content_hazard(self, item: HazardDefinition, key: str):
        return self._content[casefy.snakecase(item.__class__.__name__.replace("Definition", ""))][key]

    @content.register
    def content_string(self, item: str, key: str):
        return self._content[item][key]

    @content.register
    def content_sut(self, item: SutDescription, key: str):
        if item.key in self._content:
            return self._content[item.key][key]
        elif key == "name":
            return item.display_name
        else:
            raise ValueError(f"Unknown SUT key for {item} and {key}")

    @content.register
    def content_test(self, item: BaseTest, key: str):
        # in this case, we want to default to some sensible fallbacks
        try:
            value = self._content[item.uid][key]
        except KeyError:
            defaults = defaultdict(str)
            defaults["display_name"] = item.uid
            value = defaults[key]
        return value

    @staticmethod
    def _template_dir():
        current_path = pathlib.Path(__file__)
        while not current_path.name == "modelbench":
            current_path = current_path.parent
        return current_path / "templates"

    def _static_dir(self):
        return self._template_dir() / "static"

    def _copy_static_dir(self, output_dir):
        shutil.copytree(self._static_dir(), output_dir / "static", dirs_exist_ok=True)

    def generate(
        self,
        benchmark_runs: List[BenchmarkRun],
        output_dir: pathlib.Path,
    ) -> None:
        grouped_benchmark_runs = self._grouped_benchmark_runs(benchmark_runs)
        self._copy_static_dir(output_dir)
        self._generate_index_page(output_dir)
        self._generate_benchmarks_page(grouped_benchmark_runs, output_dir)
        self._generate_benchmark_pages(grouped_benchmark_runs, output_dir)
        self._generate_test_report_pages(grouped_benchmark_runs, output_dir)

    def _render_template(self, template_name, **kwargs):
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

    def _write_file(self, output: pathlib.Path, template_name: str, **kwargs) -> None:
        with open(pathlib.Path(output), "w+") as f:
            f.write(self._render_template(template_name, **kwargs))

    def _generate_index_page(self, output_dir: pathlib.Path) -> None:
        self._write_file(
            output=output_dir / "index.html",
            template_name="index.html",
            page_type="index",
            view_embed=self.view_embed,
        )

    @staticmethod
    def _grouped_benchmark_runs(benchmark_runs: List[BenchmarkRun]) -> Mapping[BenchmarkDefinition, List[BenchmarkRun]]:
        # TODO: Change keys to either instantiated benchmark (using .recreate_object()),
        #  or some object holding presentation-layer data.
        return group_by_key(benchmark_runs, lambda x: x.benchmark_init_record)

    def _generate_benchmarks_page(
        self, grouped_benchmark_runs: Mapping[str, List[BenchmarkRun]], output_dir: pathlib.Path
    ) -> None:
        self._write_file(
            output=output_dir / "benchmarks.html",
            template_name="benchmarks.html",
            grouped_benchmark_runs=grouped_benchmark_runs,
            show_benchmark_header=True,
            page_type="benchmarks",
            view_embed=self.view_embed,
        )

    def _generate_benchmark_pages(
        self, grouped_benchmark_runs: Mapping[str, List[BenchmarkRun]], output_dir: pathlib.Path
    ) -> None:
        for benchmark_path_name, benchmark_runs in grouped_benchmark_runs.items():
            self._write_file(
                output=output_dir / f"{benchmark_path_name}.html",
                template_name="benchmark.html",
                benchmark_path_name=benchmark_path_name,
                benchmark_runs=benchmark_runs,
                page_type="benchmark",
                view_embed=self.view_embed,
            )

    def _generate_test_report_pages(
        self, grouped_benchmark_runs: Mapping[BenchmarkDefinition, BenchmarkRun], output_dir: pathlib.Path
    ) -> None:
        for benchmark_definition in grouped_benchmark_runs:
            for run in grouped_benchmark_runs[benchmark_definition]:
                self._write_file(
                    output=output_dir / f"{run.sut.key}_{benchmark_definition.path_name()}_report.html",
                    template_name="test_report.html",
                    benchmark_definition=benchmark_definition,
                    benchmark_run=run,
                    page_type="test_report",
                    view_embed=self.view_embed,
                )

    def root_path(self) -> str:
        return "#" if self.view_embed else "index.html"

    def benchmarks_path(self, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return "../ai-safety"
        return "../../ai-safety" if self.view_embed else "benchmarks.html"

    def benchmark_path(self, benchmark_path_name, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return f"{benchmark_path_name}"
        return f"../{benchmark_path_name}" if self.view_embed else f"{benchmark_path_name}.html"

    def test_report_path(self, sut_path_name: str, benchmark_path_name: str, page_type: str) -> str:
        if page_type == "benchmarks" and self.view_embed:
            return f"{sut_path_name}_{benchmark_path_name}_report"
        return (
            f"../{sut_path_name}_{benchmark_path_name}_report"
            if self.view_embed
            else f"{sut_path_name}_{benchmark_path_name}_report.html"
        )
