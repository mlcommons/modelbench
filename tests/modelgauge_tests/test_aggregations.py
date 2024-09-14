import pytest
from modelgauge.aggregations import (
    MeasurementStats,
    get_measurement_stats,
    get_measurement_stats_by_key,
    get_measurements,
)
from modelgauge.single_turn_prompt_response import MeasuredTestItem, TestItem


def _make_measurement(measurements, context=None):
    return MeasuredTestItem(
        measurements=measurements, test_item=TestItem(prompts=[], context=context)
    )


def test_get_measurements():
    items = [
        _make_measurement({"some-key": 1}),
        _make_measurement({"some-key": 2, "another-key": 3}),
    ]
    assert get_measurements("some-key", items) == [1, 2]


def test_get_measurements_fails_missing_key():
    items = [_make_measurement({"some-key": 1}), _make_measurement({"another-key": 2})]
    with pytest.raises(KeyError):
        get_measurements("some-key", items)


def test_get_measurement_stats():
    items = [_make_measurement({"some-key": 1}), _make_measurement({"some-key": 2})]
    stats = get_measurement_stats("some-key", items)
    assert stats == MeasurementStats(
        sum=3.0, mean=1.5, count=2, population_variance=0.25, population_std_dev=0.5
    )


def test_get_measurement_stats_no_measurements():
    items = []
    stats = get_measurement_stats("some-key", items)
    assert stats == MeasurementStats(
        sum=0, mean=0, count=0, population_variance=0, population_std_dev=0
    )


def _key_by_context(item):
    return item.test_item.context


def test_get_measurement_stats_by_key():
    items = [
        _make_measurement({"some-key": 1}, context="g1"),
        _make_measurement({"some-key": 2}, context="g2"),
        _make_measurement({"some-key": 3}, context="g2"),
    ]
    stats_by_key = get_measurement_stats_by_key("some-key", items, key=_key_by_context)
    assert stats_by_key == {
        "g1": MeasurementStats(
            sum=1.0, mean=1.0, count=1, population_variance=0.0, population_std_dev=0.0
        ),
        "g2": MeasurementStats(
            sum=5.0, mean=2.5, count=2, population_variance=0.25, population_std_dev=0.5
        ),
    }
