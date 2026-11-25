from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

INF = float("inf")
MAX_EXACT_DP_GROUPS = 16


@dataclass(frozen=True)
class PrefillIOScheduleProblem:
    block_count: int
    frontiers: tuple[int, ...]
    layer_weight_io: float
    compute_base: float = 1.0
    compute_slope: float = 0.0
    layer_group_size: int = 1
    layer_groups: tuple[tuple[int, int], ...] | None = None

    @classmethod
    def from_frontiers(
        cls,
        *,
        block_count: int,
        frontiers: Iterable[int],
        layer_weight_io: float,
        compute_base: float = 1.0,
        compute_slope: float = 0.0,
        layer_group_size: int = 1,
        layer_groups: Iterable[tuple[int, int]] | None = None,
    ) -> PrefillIOScheduleProblem:
        return cls(
            block_count=int(block_count),
            frontiers=tuple(int(value) for value in frontiers),
            layer_weight_io=float(layer_weight_io),
            compute_base=float(compute_base),
            compute_slope=float(compute_slope),
            layer_group_size=int(layer_group_size),
            layer_groups=None if layer_groups is None else tuple((int(start), int(end)) for start, end in layer_groups),
        )

    @property
    def layer_count(self) -> int:
        return len(self.frontiers)

    @property
    def groups(self) -> tuple[tuple[int, int], ...]:
        if self.layer_groups is not None:
            return self.layer_groups
        return tuple(fixed_layer_groups(self.layer_count, self.layer_group_size))

    @property
    def group_count(self) -> int:
        return len(self.groups)

    @property
    def full_mask(self) -> int:
        return (1 << self.group_count) - 1

    @property
    def effective_frontiers(self) -> tuple[int, ...]:
        return tuple(layer_group_frontiers(list(self.frontiers), list(self.groups)))

    def validate(self) -> None:
        assert self.block_count >= 0
        assert self.layer_count > 0
        assert self.layer_group_size > 0
        assert self.layer_weight_io >= 0
        assert self.compute_base > 0
        assert self.compute_slope >= 0

        previous = self.block_count
        for frontier in self.frontiers:
            assert 0 <= frontier <= previous
            previous = frontier

        expected_start = 0
        for start, end in self.groups:
            assert start == expected_start
            assert start < end <= self.layer_count
            expected_start = end
        assert expected_start == self.layer_count

    def phi(self, block_count: int) -> float:
        return self.compute_base * block_count + self.compute_slope * block_count * (block_count + 1) / 2

    def group_bounds(self, group: int) -> tuple[int, int]:
        start, end = self.groups[group - 1]
        return start + 1, end

    def group_io_cost(self, group: int) -> float:
        start, end = self.groups[group - 1]
        frontiers = self.effective_frontiers
        return sum(frontiers[layer] + self.layer_weight_io for layer in range(start, end))

    def frontier_at(self, layer: int) -> int:
        if layer == 0:
            return self.block_count
        return self.effective_frontiers[layer - 1]

    def io_prefix_costs(self) -> list[float]:
        costs = [0.0] * (1 << self.group_count)
        for mask in range(1, 1 << self.group_count):
            bit = mask & -mask
            group = bit.bit_length()
            costs[mask] = costs[mask ^ bit] + self.group_io_cost(group)
        return costs

    def released_compute(self, mask: int) -> float:
        total = 0.0
        layer = 1
        while layer <= self.layer_count:
            if not self.group_ready(mask, layer):
                layer += 1
                continue

            upper_frontier = self.frontier_at(layer - 1)
            while layer <= self.layer_count and self.group_ready(mask, layer):
                total += self.phi(upper_frontier) - self.phi(self.frontier_at(layer))
                layer += 1
        return total

    def released_compute_table(self) -> list[float]:
        return [self.released_compute(mask) for mask in range(1 << self.group_count)]

    def group_ready(self, mask: int, layer: int) -> bool:
        return bool(mask & (1 << self.group_index(layer)))

    def group_index(self, layer: int) -> int:
        zero_based_layer = layer - 1
        for group, (start, end) in enumerate(self.groups):
            if start <= zero_based_layer < end:
                return group
        raise AssertionError(layer)


@dataclass(frozen=True)
class PrefillIOScheduleResult:
    order: tuple[int, ...]
    exposed_io: float
    total_compute: float
    total_time: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["order"] = list(self.order)
        return payload


def fixed_layer_groups(layer_count: int, group_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + group_size, layer_count)) for start in range(0, layer_count, group_size)]


def frontier_groups(frontiers: list[int]) -> list[tuple[int, int]]:
    groups = []
    start = 0
    while start < len(frontiers):
        end = start + 1
        while end < len(frontiers) and frontiers[end] == frontiers[start]:
            end += 1
        groups.append((start, end))
        start = end
    return groups


def layer_group_frontiers(frontiers: list[int], groups: list[tuple[int, int]]) -> list[int]:
    grouped = list(frontiers)
    expected_start = 0
    for start, end in groups:
        assert start == expected_start
        assert start < end <= len(frontiers)
        group_frontier = min(frontiers[start:end])
        for layer in range(start, end):
            grouped[layer] = group_frontier
        expected_start = end
    assert expected_start == len(frontiers)
    return grouped


def evaluate_order(problem: PrefillIOScheduleProblem, order: Iterable[int]) -> PrefillIOScheduleResult:
    problem.validate()
    io_costs = problem.io_prefix_costs()
    compute = problem.released_compute_table()
    mask = 0
    exposed_io = 0.0
    normalized_order = tuple(int(group) for group in order)
    assert sorted(normalized_order) == list(range(1, problem.group_count + 1))

    for group in normalized_order:
        exposed_io = max(exposed_io, io_costs[mask] + problem.group_io_cost(group) - compute[mask])
        mask |= 1 << (group - 1)

    exposed_io = max(exposed_io, 0.0)
    total_compute = compute[problem.full_mask]
    return PrefillIOScheduleResult(
        order=normalized_order,
        exposed_io=exposed_io,
        total_compute=total_compute,
        total_time=total_compute + exposed_io,
    )


def solve_state_compression_dp(problem: PrefillIOScheduleProblem) -> PrefillIOScheduleResult:
    problem.validate()
    io_costs = problem.io_prefix_costs()
    compute = problem.released_compute_table()
    state_count = 1 << problem.group_count
    dp = [INF] * state_count
    parent: list[tuple[int, int] | None] = [None] * state_count
    dp[0] = 0.0

    for mask in range(state_count):
        if dp[mask] == INF:
            continue
        for group in range(1, problem.group_count + 1):
            bit = 1 << (group - 1)
            if mask & bit:
                continue
            new_mask = mask | bit
            exposed_if_added = io_costs[mask] + problem.group_io_cost(group) - compute[mask]
            candidate = max(dp[mask], exposed_if_added)
            if candidate < dp[new_mask]:
                dp[new_mask] = candidate
                parent[new_mask] = (mask, group)

    order = []
    mask = problem.full_mask
    while mask:
        previous_mask, group = parent[mask]
        order.append(group)
        mask = previous_mask
    order.reverse()

    exposed_io = max(dp[problem.full_mask], 0.0)
    total_compute = compute[problem.full_mask]
    return PrefillIOScheduleResult(
        order=tuple(order),
        exposed_io=exposed_io,
        total_compute=total_compute,
        total_time=total_compute + exposed_io,
    )


def group_order_baseline(problem: PrefillIOScheduleProblem) -> PrefillIOScheduleResult:
    return evaluate_order(problem, range(1, problem.group_count + 1))


def state_compression_problem_from_frontiers(
    frontiers: Iterable[int],
    *,
    block_count: int,
    layer_weight_io: float | None = None,
    compute_base: float = 1.0,
    compute_slope: float = 0.0,
    layer_group_size: int = 1,
    layer_groups: Iterable[tuple[int, int]] | None = None,
) -> PrefillIOScheduleProblem:
    return PrefillIOScheduleProblem.from_frontiers(
        block_count=block_count,
        frontiers=frontiers,
        layer_weight_io=block_count if layer_weight_io is None else layer_weight_io,
        compute_base=compute_base,
        compute_slope=compute_slope,
        layer_group_size=layer_group_size,
        layer_groups=layer_groups,
    )


def state_compression_group_order(
    frontiers: list[int],
    *,
    block_count: int,
    layer_weight_io: float | None = None,
    layer_group_size: int = 1,
    layer_groups: list[tuple[int, int]] | None = None,
    max_exact_groups: int | None = MAX_EXACT_DP_GROUPS,
) -> tuple[list[int], PrefillIOScheduleProblem, PrefillIOScheduleResult | None]:
    problem = state_compression_problem_from_frontiers(
        frontiers,
        block_count=block_count,
        layer_weight_io=layer_weight_io,
        layer_group_size=layer_group_size,
        layer_groups=layer_groups,
    )
    if max_exact_groups is not None and problem.group_count > max_exact_groups:
        return list(range(problem.group_count)), problem, None
    result = solve_state_compression_dp(problem)
    return [group - 1 for group in result.order], problem, result


def state_compression_layer_order(
    frontiers: list[int],
    *,
    block_count: int,
    layer_weight_io: float | None = None,
    layer_group_size: int = 1,
    max_exact_groups: int | None = MAX_EXACT_DP_GROUPS,
    group_equal_frontiers: bool = False,
) -> list[int]:
    layer_groups = fixed_layer_groups(len(frontiers), layer_group_size)
    if (
        group_equal_frontiers
        and layer_group_size == 1
        and max_exact_groups is not None
        and len(frontiers) > max_exact_groups
    ):
        layer_groups = frontier_groups(frontiers)

    group_order, problem, _ = state_compression_group_order(
        frontiers,
        block_count=block_count,
        layer_weight_io=layer_weight_io,
        layer_group_size=layer_group_size,
        layer_groups=layer_groups,
        max_exact_groups=max_exact_groups,
    )
    if max_exact_groups is not None and len(group_order) != problem.group_count:
        return list(range(len(frontiers)))
    return [layer for group in group_order for layer in range(problem.groups[group][0], problem.groups[group][1])]
