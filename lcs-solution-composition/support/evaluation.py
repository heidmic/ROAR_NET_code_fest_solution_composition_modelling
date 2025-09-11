import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Rule:
    local_predictions: dict[int, float]  # indices present => matches those samples
    tau: float

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Rule":
        lp = {int(k): float(v) for k, v in d["local_predictions"].items()}
        return Rule(local_predictions=lp, tau=float(d["tau"]))


class Problem:
    def __init__(self, alpha: float, target_values: list[float], rule_pool: list[Rule]):
        if not target_values or not rule_pool:
            raise ValueError("Problem needs at least one target and one rule.")

        self.alpha = float(alpha)
        self.y = list(map(float, target_values))
        self.rules = list(rule_pool)
        self.n, self.k = len(self.y), len(self.rules)
        self._y_mean = sum(self.y) / self.n

        # Basic index validation
        for i, r in enumerate(self.rules):
            bad = [j for j in r.local_predictions.keys() if not (0 <= j < self.n)]
            if bad:
                raise ValueError(f"Rule {i} has out-of-range indices: {bad}")

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Problem":
        rules = [Rule.from_dict(r) for r in d["rule_pool"]]
        return Problem(alpha=d.get("alpha", 0.3), target_values=d["target_values"], rule_pool=rules)

    @staticmethod
    def from_json(path: str) -> "Problem":
        with open(path, "r") as f:
            return Problem.from_dict(json.load(f))

    def predict(self, rule_set: set[int]) -> list[float]:
        preds = []
        for j in range(self.n):
            num = den = 0.0
            for i in rule_set:
                r = self.rules[i]
                yij = r.local_predictions.get(j)
                if yij is not None:
                    den += r.tau
                    num += r.tau * yij
            preds.append(self._y_mean if den == 0.0 else num / den)
        return preds

    def performance(self, rule_set: set[int]) -> float:
        mse = sum((yt - ph) ** 2 for yt, ph in zip(self.y, self.predict(rule_set))) / self.n
        return math.exp(-2.0 * mse)

    def compactness(self, rule_set: set[int]) -> float:
        return 1.0 - (len(rule_set) / self.k)


class Solution:
    def __init__(self, problem: Problem, indices: list[int] | None = None, mask: list[bool] | None = None):

        if indices is None and mask is None:
            raise ValueError("Either indices or mask must be provided.")

        self.problem = problem

        if mask is not None:
            indices = [i for i, value in enumerate(mask) if value]

        self.rule_set = set(indices)

    @staticmethod
    def from_json(path: str, problem: Problem) -> "Solution":
        with open(path, "r") as f:
            rule_set = json.load(f)['rule_set']

            if len(rule_set) == problem.k and all(map(lambda x: x in (0, 1) or isinstance(x, bool), rule_set)):
                return Solution(problem, mask=rule_set)
            else:
                return Solution(problem, indices=rule_set)

    def objective_value(self) -> float:
        a = self.problem.alpha
        pa = self.problem.performance(self.rule_set)
        cn = self.problem.compactness(self.rule_set)
        denom = (a * a) * pa + cn
        return 0.0 if denom == 0.0 else ((1 + a * a) * pa * cn) / denom


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        prob = Problem.from_json(sys.argv[1])
        sol = Solution.from_json(sys.argv[2], prob)
        print(f"{sol.objective_value():.12f}")