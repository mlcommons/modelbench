from typing import Dict, List
from newhelm.placeholders import Measurement, Result
from newhelm.single_turn_prompt_response import MeasuredTestItem


def get_measurements(
    measurement_name: str, items: List[MeasuredTestItem]
) -> List[Measurement]:
    """Extract a desired measurement for all TestItems."""
    # TODO: Consider adding asserts that all test items have exactly one
    # measurement with this name.
    measurements = []
    for test_item in items:
        for measurement in test_item.measurements:
            if measurement.name == measurement_name:
                measurements.append(measurement)
    return measurements


def mean_of_measurement(measurement_name: str, items: List[MeasuredTestItem]) -> float:
    """Calculate the mean across all TestItems for a desired measurement."""
    measurements = get_measurements(measurement_name, items)
    total = sum(measurement.value for measurement in measurements)
    return total / len(measurements)


def mean_of_results(results: Dict[str, List[Result]]) -> float:
    """Calculate the mean across all Results from all Tests."""
    flattened = sum(results.values(), [])
    if len(flattened) == 0:
        return 0
    total = sum(r.value for r in flattened)
    return total / len(flattened)
