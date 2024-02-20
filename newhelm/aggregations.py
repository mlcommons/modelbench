import math
from typing import Dict, List

from newhelm.base_test import Result
from newhelm.single_turn_prompt_response import MeasuredTestItem


def get_measurements(
    measurement_name: str, items: List[MeasuredTestItem]
) -> List[float]:
    """Extract a desired measurement for all TestItems."""
    # Raises a KeyError if that test item is missing that measurement.
    return [item.measurements[measurement_name] for item in items]


def sum_measurements(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    measurements = get_measurements(measurement_name, items)
    return sum(measurement for measurement in measurements)


def mean_of_measurement(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    """Calculate the mean across all TestItems for a desired measurement."""
    measurements = get_measurements(measurement_name, items)
    total = sum(measurements)
    return total / len(measurements)


def variance_of_measurement(
    measurement_name: str, items: List[MeasuredTestItem]
) -> float:
    """Calculate the variance across all TestItems for a desired measurement."""
    measurements = get_measurements(measurement_name, items)
    mean = mean_of_measurement(measurement_name, items)
    deviations = [(x - mean) ** 2 for x in measurements]
    variance = sum(deviations) / len(measurements)
    return variance


def stddev_of_measurement(
    measurement_name: str, items: List[MeasuredTestItem]
) -> float:
    """Calculate the standard deviation across all TestItems for a desired measurement."""
    variance = variance_of_measurement(measurement_name, items)
    return math.sqrt(variance)


def mean_of_results(results: Dict[str, List[Result]]) -> float:
    """Calculate the mean across all Results from all Tests."""
    flattened = sum(results.values(), [])
    if len(flattened) == 0:
        return 0
    total = sum(r.value for r in flattened)
    return total / len(flattened)
