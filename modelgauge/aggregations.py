import math
from collections import defaultdict
from modelgauge.single_turn_prompt_response import MeasuredTestItem
from pydantic import BaseModel
from typing import Callable, List, Mapping, Sequence, TypeVar


def get_measurements(
    measurement_name: str, items: List[MeasuredTestItem]
) -> List[float]:
    """Extract a desired measurement for all TestItems."""
    # Raises a KeyError if that test item is missing that measurement.
    return [item.measurements[measurement_name] for item in items]


class MeasurementStats(BaseModel):
    """Container for common stats about a measurement."""

    sum: float
    mean: float
    count: int
    population_variance: float
    population_std_dev: float
    # TODO Consider min, max, and median

    @staticmethod
    def calculate(values: Sequence[float]) -> "MeasurementStats":
        if len(values) == 0:
            return MeasurementStats(
                sum=0, mean=0, count=0, population_variance=0, population_std_dev=0
            )
        total = sum(values)
        count = len(values)
        mean = total / count
        deviations = [(x - mean) ** 2 for x in values]
        variance = sum(deviations) / len(values)
        std_dev = math.sqrt(variance)
        return MeasurementStats(
            sum=total,
            mean=mean,
            count=count,
            population_variance=variance,
            population_std_dev=std_dev,
        )


def get_measurement_stats(
    measurement_name: str, items: List[MeasuredTestItem]
) -> MeasurementStats:
    """Calculate common statistics about `measurement_name`."""
    values = get_measurements(measurement_name, items)
    return MeasurementStats.calculate(values)


_T = TypeVar("_T")


def get_measurement_stats_by_key(
    measurement_name: str,
    items: List[MeasuredTestItem],
    *,
    key: Callable[[MeasuredTestItem], _T]
) -> Mapping[_T, MeasurementStats]:
    """Calculate statistics grouping by `key`."""
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    stats = {}
    for key_value, measurements in groups.items():
        stats[key_value] = get_measurement_stats(measurement_name, measurements)
    return stats


def sum_measurements(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    measurements = get_measurements(measurement_name, items)
    return sum(measurement for measurement in measurements)


def mean_of_measurement(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    """Calculate the mean across all TestItems for a desired measurement."""
    measurements = get_measurements(measurement_name, items)
    total = sum(measurements)
    return total / len(measurements)
