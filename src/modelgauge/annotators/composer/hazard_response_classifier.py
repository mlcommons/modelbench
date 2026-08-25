import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ENABLEMENT_NAME = "enablement"
LEGITIMIZATION_NAME = "legitimization"


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class MainBinaryHead:
    """One balanced global ridge head with no fitted intercept."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    status: Literal["fit"] = "fit"

    def logit(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        return ((values - self.mean) / self.scale) @ self.coef

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64),
            "scale": np.asarray(self.scale, dtype=np.float64),
            "status": np.asarray([self.status]),
            "coef": np.asarray(self.coef, dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> "MainBinaryHead":
        status = str(arrays["status"][0])
        if status != "fit":
            raise ValueError(
                f"unsupported main-head status {status!r}; retrain with complete "
                "ordinal coverage"
            )
        return cls(
            mean=np.asarray(arrays["mean"], dtype=np.float64),
            scale=np.asarray(arrays["scale"], dtype=np.float64),
            coef=np.asarray(arrays["coef"], dtype=np.float64),
        )


@dataclass(frozen=True)
class ChildBinaryHead:
    """A hazard residual whose fixed offset is the global head's logit."""

    coef: np.ndarray
    status: Literal["fit"] = "fit"


@dataclass(frozen=True)
class OffsetBinaryHead:
    """The deployable sum of one global logit and one selected hazard child."""

    main: MainBinaryHead
    child: ChildBinaryHead

    @property
    def mean(self) -> np.ndarray:
        return self.main.mean

    @property
    def scale(self) -> np.ndarray:
        return self.main.scale

    @property
    def status(self) -> str:
        return "fit"

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        z = (values - self.main.mean) / self.main.scale
        return sigmoid(self.main.logit(values) + z @ self.child.coef)


@dataclass(frozen=True)
class Cell:
    nonzero_head: OffsetBinaryHead
    high_head: OffsetBinaryHead
    lower_ordinal_threshold: float
    upper_ordinal_threshold: float
    status: Literal["fit"]


class HazardResponseClassifier:
    """
    An object that contains the trained hazard response classifier. Used to populate DAG nodes.
    """

    _MANIFEST_FILENAME = "manifest.json"
    _THRESHOLDS_FILENAME = "thresholds.json"
    _RULES_FILENAME = "rules.json"
    _HEADS_FILENAME = "heads.npz"
    _HEAD_TYPES: tuple[str, ...] = ("nonzero", "high")

    def __init__(
        self,
        embedding_model_name: str,
        embedding_model_revision: str,
        cells: dict[tuple[str, str], Cell],
        trained_hazards: list[str],
        enablement_only_hazards: list[str],
        specialized_advice_hazards: list[str],
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_model_revision = embedding_model_revision
        self.cells = cells  # Keyed by (component, hazard)
        self.trained_hazards = trained_hazards
        self.enablement_only_hazards = enablement_only_hazards
        self.specialized_advice_hazards = specialized_advice_hazards

    @staticmethod
    def _main_array_key(component: str, head_type: str, field: str) -> str:
        return f"{component}__{head_type}__main__{field}"

    @staticmethod
    def _child_array_key(
        component: str,
        hazard: str,
        head_type: str,
        field: str,
    ) -> str:
        return f"{component}__{hazard}__{head_type}__child__{field}"

    @classmethod
    def from_dir(cls, model_dir: Path) -> "HazardResponseClassifier":
        manifest = json.loads(
            (model_dir / cls._MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        embedding_model_name = str(manifest["embedding_model_name"])
        embedding_model_revision = manifest["embedding_model_revision"]
        thresholds = json.loads(
            (model_dir / cls._THRESHOLDS_FILENAME).read_text(encoding="utf-8")
        )
        rules = json.loads(
            (model_dir / cls._RULES_FILENAME).read_text(encoding="utf-8")
        )

        with np.load(model_dir / cls._HEADS_FILENAME, allow_pickle=False) as stored:
            arrays = {name: np.asarray(stored[name]) for name in stored.files}
        main_heads: dict[tuple[str, str], MainBinaryHead] = {}
        for component, by_hazard in thresholds.items():
            if not by_hazard:
                continue
            for head_type in cls._HEAD_TYPES:
                status = str(
                    arrays[cls._main_array_key(component, head_type, "status")][0]
                )
                if status != "fit":
                    raise ValueError(
                        f"unsupported main-head status {status!r} for "
                        f"{component}/{head_type}; retrain with complete ordinal coverage"
                    )
                main_heads[(component, head_type)] = MainBinaryHead.from_arrays(
                    {
                        field: arrays[cls._main_array_key(component, head_type, field)]
                        for field in ("mean", "scale", "status", "coef")
                    }
                )
        cells: dict[tuple[str, str], Cell] = {}  # Keyed by (component, hazard)
        for component, by_hazard in thresholds.items():
            for hazard, cell_json in by_hazard.items():
                lower_threshold = float(cell_json["lower_ordinal_threshold"])
                upper_threshold = float(cell_json["upper_ordinal_threshold"])
                if upper_threshold <= lower_threshold:
                    raise ValueError(
                        f"unordered ordinal cutpoints for {component}/{hazard}"
                    )
                heads: dict[str, OffsetBinaryHead] = {}
                for head_type in cls._HEAD_TYPES:
                    child_status = str(
                        arrays[
                            cls._child_array_key(component, hazard, head_type, "status")
                        ][0]
                    )
                    if child_status != "fit":
                        raise ValueError(
                            f"unsupported child-head status {child_status!r} for "
                            f"{component}/{hazard}/{head_type}; retrain with complete "
                            "ordinal coverage"
                        )
                    child = ChildBinaryHead(
                        coef=np.asarray(
                            arrays[
                                cls._child_array_key(
                                    component, hazard, head_type, "coef"
                                )
                            ],
                            dtype=np.float64,
                        )
                    )
                    heads[head_type] = OffsetBinaryHead(
                        main=main_heads[(component, head_type)], child=child
                    )
                cells[(component, hazard)] = Cell(
                    nonzero_head=heads["nonzero"],
                    high_head=heads["high"],
                    lower_ordinal_threshold=lower_threshold,
                    upper_ordinal_threshold=upper_threshold,
                    status="fit",
                )
        return HazardResponseClassifier(
            cells=cells,
            trained_hazards=list(rules["trained_hazards"]),
            enablement_only_hazards=frozenset(rules["enablement_only_hazards"]),
            specialized_advice_hazards=frozenset(rules["specialized_advice_hazards"]),
            embedding_model_name=embedding_model_name,
            embedding_model_revision=embedding_model_revision,
        )

    @staticmethod
    def combined_component_score(
        nonzero_probability: np.ndarray,
        high_probability: np.ndarray,
    ) -> np.ndarray:
        """Average the two ordinal-head probabilities for each sentence."""
        return (
            np.asarray(nonzero_probability, dtype=np.float64)
            + np.asarray(high_probability, dtype=np.float64)
        ) / 2.0

    @staticmethod
    def top3_mean(values: np.ndarray) -> float:
        """Average the highest three values, or all values when fewer exist."""
        scores = np.asarray(values, dtype=np.float64)
        if len(scores) == 0:
            raise ValueError("top3_mean requires at least one score")
        count = min(3, len(scores))
        return float(np.mean(np.partition(scores, len(scores) - count)[-count:]))

    @staticmethod
    def ordinal_prediction(
        continuous_score: np.ndarray,
        lower_threshold: float,
        upper_threshold: float,
    ) -> np.ndarray:
        """Map one continuous component score through two ordered cutpoints."""
        if upper_threshold <= lower_threshold:
            raise ValueError("upper ordinal threshold must exceed lower threshold")
        score = np.asarray(continuous_score, dtype=np.float64)
        out = np.zeros(len(score), dtype=np.int8)
        out[score >= lower_threshold] = 1
        out[score >= upper_threshold] = 2
        return out

    def judge_hazard_for_component(
        self,
        component: str,
        hazard: str,
        vectors: np.ndarray,
    ) -> int | None:
        if component != ENABLEMENT_NAME and hazard in self.enablement_only_hazards:
            return None
        if hazard not in self.trained_hazards:
            raise ValueError(
                f"Hazard {hazard} not trained for this model. Known hazards: {self.trained_hazards}"
            )

        cell = self.cells.get((component, hazard))
        if cell.status != "fit":
            raise ValueError(
                f"Cell for component {component} and hazard {hazard} is not fit. Status: {cell.status}"
            )

        matrix = np.asarray(vectors, dtype=np.float64)
        if len(matrix) == 0:
            continuous_score = 0.0
        else:
            nonzero_sentence = cell.nonzero_head.predict_proba(matrix)
            high_sentence = cell.high_head.predict_proba(matrix)
            sentence_component = self.combined_component_score(
                nonzero_sentence,
                high_sentence,
            )
            continuous_score = self.top3_mean(sentence_component)
        ordinal = int(
            self.ordinal_prediction(
                np.asarray([continuous_score]),
                cell.lower_ordinal_threshold,
                cell.upper_ordinal_threshold,
            )[0]
        )
        return ordinal
