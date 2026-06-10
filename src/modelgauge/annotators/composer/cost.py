from dataclasses import dataclass


@dataclass
class CostInfo:
    input_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    fixed_cost: float = 0.0
    latency_seconds: float = 0.0

    def __add__(self, other: "CostInfo") -> "CostInfo":
        return CostInfo(
            input_cost_per_token=self.input_cost_per_token + other.input_cost_per_token,
            output_cost_per_token=self.output_cost_per_token
            + other.output_cost_per_token,
            fixed_cost=self.fixed_cost + other.fixed_cost,
            latency_seconds=self.latency_seconds + other.latency_seconds,
        )


@dataclass
class RealizedCost:
    input_token_cost: float = 0.0
    output_token_cost: float = 0.0
    fixed_cost: float = 0.0
    latency_seconds: float = 0.0

    @property
    def total_token_cost(self) -> float:
        return self.input_token_cost + self.output_token_cost

    @property
    def total_cost(self) -> float:
        return self.total_token_cost + self.fixed_cost

    def __add__(self, other: "RealizedCost") -> "RealizedCost":
        return RealizedCost(
            input_token_cost=self.input_token_cost + other.input_token_cost,
            output_token_cost=self.output_token_cost + other.output_token_cost,
            fixed_cost=self.fixed_cost + other.fixed_cost,
            latency_seconds=self.latency_seconds + other.latency_seconds,
        )

    def to_dict(self) -> dict:
        return {
            "input_token_cost": self.input_token_cost,
            "output_token_cost": self.output_token_cost,
            "fixed_cost": self.fixed_cost,
            "latency_seconds": self.latency_seconds,
            "total_token_cost": self.total_token_cost,
            "total_cost": self.total_cost,
        }
