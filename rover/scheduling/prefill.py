from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import torch

from rover.runtime.types import ChunkLayerIndex, PrefillInputs, Range
from rover.scheduling.dp import (
    MAX_EXACT_DP_GROUPS,
    fixed_layer_groups,
    frontier_groups,
    layer_group_frontiers,
    state_compression_group_order,
    state_compression_layer_order,
)

MAX_EXACT_DP_LAYERS = MAX_EXACT_DP_GROUPS


if TYPE_CHECKING:
    from rover.runtime.runner import GenerationRunner


class Resource(Enum):
    IO = "io"
    COMPUTE = "compute"


class TaskKind(Enum):
    READ_PARAMS = "read_params"
    COMPUTE_CELL = "compute_cell"
    READ_KV = "read_kv"
    WRITE_KV = "write_kv"
    READ_H = "read_h"
    WRITE_H = "write_h"


@dataclass(frozen=True, order=True)
class Cell:
    chunk: int
    layer: int
    start: int
    end: int

    @property
    def range(self) -> Range:
        return Range(self.layer, self.start, self.end)

    @property
    def index(self) -> ChunkLayerIndex:
        return ChunkLayerIndex(self.chunk, self.layer, self.start, self.end)


@dataclass(frozen=True)
class Task:
    id: str
    kind: TaskKind
    resource: Resource
    cell: Cell | None = None
    layer: int | None = None


@dataclass(frozen=True)
class ScheduleSnapshot:
    chunk_count: int
    layer_count: int
    layer_group_size: int
    completed_tasks: frozenset[str]
    cell_done: frozenset[Cell]
    resident_kv: frozenset[Cell]
    resident_h: frozenset[Cell]
    resident_params: frozenset[int]
    persisted_kv: frozenset[Cell]
    persisted_h: frozenset[Cell]
    running_io_task: Task | None
    running_compute_task: Task | None


class PrefillScheduleStrategy(Protocol):
    def choose_io(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None: ...

    def choose_compute(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None: ...


class ChunkFirstStrategy:
    def choose_io(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None:
        return _first_by_priority(ready, _io_priority)

    def choose_compute(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None:
        return _first_by_priority(ready, lambda task: (task.cell.chunk, task.cell.layer))


class LayerFirstStrategy:
    def choose_io(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None:
        return _first_by_priority(ready, _io_priority)

    def choose_compute(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None:
        return _first_by_priority(ready, lambda task: (task.cell.layer, task.cell.chunk))


class GreedyIOComputeStrategy(ChunkFirstStrategy):
    pass


class StateCompressionDPStrategy(ChunkFirstStrategy):
    def choose_io(self, snapshot: ScheduleSnapshot, ready: list[Task]) -> Task | None:
        if not ready:
            return None

        writes = [task for task in ready if task.kind in {TaskKind.WRITE_KV, TaskKind.WRITE_H}]
        if writes:
            return _first_by_priority(writes, _io_priority)

        layer_rank = _dp_layer_rank(snapshot)
        return _first_by_priority(ready, lambda task: _dp_io_priority(task, layer_rank))


def build_prefill_strategy(name: str) -> PrefillScheduleStrategy:
    strategies = {
        "chunk_first": ChunkFirstStrategy,
        "layer_first": LayerFirstStrategy,
        "greedy": GreedyIOComputeStrategy,
        "state_compression_dp": StateCompressionDPStrategy,
    }
    assert name in strategies
    return strategies[name]()


def _first_by_priority(ready: list[Task], key):
    if not ready:
        return None
    return min(ready, key=key)


def _io_priority(task: Task):
    kind_order = {
        TaskKind.WRITE_KV: 0,
        TaskKind.WRITE_H: 1,
        TaskKind.READ_PARAMS: 2,
        TaskKind.READ_KV: 3,
        TaskKind.READ_H: 4,
    }
    cell_key = (-1, task.layer if task.layer is not None else -1)
    if task.cell is not None:
        cell_key = (task.cell.chunk, task.cell.layer)
    return kind_order[task.kind], cell_key


def _dp_io_priority(task: Task, layer_rank: dict[int, int]):
    kind_order = {
        TaskKind.READ_PARAMS: 0,
        TaskKind.READ_KV: 1,
        TaskKind.READ_H: 2,
    }
    layer = task.layer if task.layer is not None else task.cell.layer
    cell_key = (-1, -1) if task.cell is None else (task.cell.chunk, task.cell.layer)
    return layer_rank[layer], kind_order[task.kind], cell_key


def _dp_layer_rank(snapshot: ScheduleSnapshot) -> dict[int, int]:
    order = _state_compression_dp_order(
        _frontiers_from_snapshot(snapshot), snapshot.chunk_count, snapshot.layer_group_size
    )
    return {layer: rank for rank, layer in enumerate(order)}


def _frontiers_from_snapshot(snapshot: ScheduleSnapshot) -> list[int]:
    frontiers = []
    for layer in range(snapshot.layer_count):
        done_chunks = [
            cell.chunk for cell in snapshot.persisted_kv if cell.layer == layer and cell in snapshot.persisted_h
        ]
        frontier = 0
        while frontier in done_chunks:
            frontier += 1
        frontiers.append(frontier)
    return frontiers


def _state_compression_dp_order(
    frontiers: list[int],
    chunk_count: int,
    layer_group_size: int = 1,
    layer_groups: list[tuple[int, int]] | None = None,
) -> list[int]:
    if layer_groups is not None:
        group_order, problem, _ = state_compression_group_order(
            frontiers,
            block_count=chunk_count,
            layer_weight_io=chunk_count,
            layer_group_size=layer_group_size,
            layer_groups=layer_groups,
            max_exact_groups=MAX_EXACT_DP_LAYERS,
        )
        return [layer for group in group_order for layer in range(problem.groups[group][0], problem.groups[group][1])]

    return state_compression_layer_order(
        frontiers,
        block_count=chunk_count,
        layer_weight_io=chunk_count,
        layer_group_size=layer_group_size,
        max_exact_groups=MAX_EXACT_DP_LAYERS,
        group_equal_frontiers=True,
    )


def _fixed_layer_groups(layer_count: int, group_size: int) -> list[tuple[int, int]]:
    return fixed_layer_groups(layer_count, group_size)


def _layer_group_frontiers(frontiers: list[int], groups: list[tuple[int, int]]) -> list[int]:
    return layer_group_frontiers(frontiers, groups)


def _frontier_groups(frontiers: list[int]) -> list[tuple[int, int]]:
    return frontier_groups(frontiers)


def _state_compression_dp_group_order(
    frontiers: list[int],
    chunk_count: int,
    groups: list[tuple[int, int]],
) -> list[int]:
    group_count = len(groups)
    group_layer_masks = [_group_layer_mask(group) for group in groups]
    io_costs = _group_io_cost_table(frontiers, chunk_count, groups)
    compute = _group_compute_table(frontiers, chunk_count, group_layer_masks)
    state_count = 1 << group_count
    dp = [float("inf")] * state_count
    parent: list[tuple[int, int] | None] = [None] * state_count
    dp[0] = 0.0

    for mask in range(state_count):
        if dp[mask] == float("inf"):
            continue
        for group in range(group_count):
            bit = 1 << group
            if mask & bit:
                continue
            new_mask = mask | bit
            exposed = io_costs[mask] + _group_io_cost(frontiers, chunk_count, groups[group]) - compute[mask]
            candidate = max(dp[mask], exposed)
            if candidate < dp[new_mask]:
                dp[new_mask] = candidate
                parent[new_mask] = (mask, group)

    order = []
    mask = state_count - 1
    while mask:
        previous_mask, group = parent[mask]
        order.append(group)
        mask = previous_mask
    return list(reversed(order))


def _group_layer_mask(group: tuple[int, int]) -> int:
    start, end = group
    mask = 0
    for layer in range(start, end):
        mask |= 1 << layer
    return mask


def _group_io_cost(frontiers: list[int], layer_weight_io: float, group: tuple[int, int]) -> float:
    start, end = group
    return sum(frontiers[layer] + layer_weight_io for layer in range(start, end))


def _group_io_cost_table(
    frontiers: list[int],
    layer_weight_io: float,
    groups: list[tuple[int, int]],
) -> list[float]:
    costs = [0.0] * (1 << len(groups))
    for mask in range(1, 1 << len(groups)):
        bit = mask & -mask
        group = bit.bit_length() - 1
        costs[mask] = costs[mask ^ bit] + _group_io_cost(frontiers, layer_weight_io, groups[group])
    return costs


def _group_compute_table(frontiers: list[int], chunk_count: int, group_layer_masks: list[int]) -> list[float]:
    table = []
    for mask in range(1 << len(group_layer_masks)):
        layer_mask = 0
        for group, group_layer_mask in enumerate(group_layer_masks):
            if mask & (1 << group):
                layer_mask |= group_layer_mask
        table.append(_released_compute(frontiers, chunk_count, layer_mask))
    return table


def _released_compute(frontiers: list[int], chunk_count: int, layer_mask: int) -> float:
    total = 0.0
    layer = 0
    while layer < len(frontiers):
        if not layer_mask & (1 << layer):
            layer += 1
            continue
        upper_frontier = chunk_count if layer == 0 else frontiers[layer - 1]
        while layer < len(frontiers) and layer_mask & (1 << layer):
            total += upper_frontier - frontiers[layer]
            layer += 1
    return total


class PrefillScheduler:
    def __init__(
        self,
        model: GenerationRunner,
        prefill_inputs: PrefillInputs,
        strategy: PrefillScheduleStrategy,
    ):
        self.model = model
        self.state = model.state
        self.prefill_inputs = prefill_inputs
        self.strategy = strategy
        self.chunk_size = model.runtime_config.prefill_chunk_size
        self.prompt_len = prefill_inputs.inputs_embeds.shape[1]
        self.chunk_count = (self.prompt_len + self.chunk_size - 1) // self.chunk_size
        self.layer_count = model.num_layers
        self.cells = [
            Cell(chunk, layer, chunk * self.chunk_size, min((chunk + 1) * self.chunk_size, self.prompt_len))
            for layer in range(self.layer_count)
            for chunk in range(self.chunk_count)
        ]

        self.completed_tasks: set[str] = set()
        self.cell_done: set[Cell] = set()
        self.resident_kv: set[Cell] = set()
        self.resident_h: set[Cell] = set()
        self.resident_params: set[int] = set(model.loaded_layers)
        self.persisted_kv: set[Cell] = set()
        self.persisted_h: set[Cell] = set()
        self._kv_written: set[Cell] = set()
        self._h_written: set[Cell] = set()
        self._running_io_task: Task | None = None
        self._running_compute_task: Task | None = None
        self._io_future: Future[Task] | None = None
        self._compute_future: Future[Task] | None = None
        self._seed_persisted_progress()

    def run(self) -> torch.Tensor:
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rover-prefill-io") as io_pool,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="rover-prefill-compute") as compute_pool,
        ):
            while not self._complete:
                self._dispatch_ready(io_pool, compute_pool)
                futures = [f for f in (self._io_future, self._compute_future) if f is not None]
                assert futures, "prefill scheduler stalled with no runnable tasks"
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    self._finish(future.result())
        final_cell = self._cell(self.chunk_count - 1, self.layer_count - 1)
        final = self.state.get_h(final_cell.range)[:, -1:, :]
        return self.model.model.norm(final)

    @property
    def _complete(self) -> bool:
        return all(cell in self.persisted_kv and cell in self.persisted_h for cell in self.cells)

    def _seed_persisted_progress(self) -> None:
        for layer, chunks_done in enumerate(self.state.layer_progress):
            for chunk in range(chunks_done):
                cell = self._cell(chunk, layer)
                self.cell_done.add(cell)
                self.resident_kv.add(cell)
                self.persisted_kv.add(cell)
                self.persisted_h.add(cell)
                self._kv_written.add(cell)
                self._h_written.add(cell)
                self.completed_tasks.update(
                    {
                        self._task_id(TaskKind.COMPUTE_CELL, cell),
                        self._task_id(TaskKind.WRITE_KV, cell),
                        self._task_id(TaskKind.WRITE_H, cell),
                    }
                )

    def _dispatch_ready(self, io_pool: ThreadPoolExecutor, compute_pool: ThreadPoolExecutor) -> None:
        snapshot = self._snapshot()
        if self._io_future is None:
            task = self.strategy.choose_io(snapshot, self._ready_io_tasks())
            if task is not None:
                self._running_io_task = task
                self._io_future = io_pool.submit(self._run_task, task)
        if self._compute_future is None:
            task = self.strategy.choose_compute(snapshot, self._ready_compute_tasks())
            if task is not None:
                self._running_compute_task = task
                self._compute_future = compute_pool.submit(self._run_task, task)

    def _finish(self, task: Task) -> None:
        self.completed_tasks.add(task.id)
        if task.resource == Resource.IO:
            self._io_future = None
            self._running_io_task = None
        else:
            self._compute_future = None
            self._running_compute_task = None

        if task.kind == TaskKind.READ_PARAMS:
            self.resident_params.add(task.layer)
        elif task.kind == TaskKind.COMPUTE_CELL:
            self.cell_done.add(task.cell)
            self.resident_kv.add(task.cell)
            self.resident_h.add(task.cell)
            if task.cell.layer > 0:
                previous = self._cell(task.cell.chunk, task.cell.layer - 1)
                if previous in self.persisted_h:
                    self.state.drop_h(previous.range)
                    self.resident_h.discard(previous)
        elif task.kind == TaskKind.READ_KV:
            self.resident_kv.add(task.cell)
        elif task.kind == TaskKind.READ_H:
            self.resident_h.add(task.cell)
        elif task.kind == TaskKind.WRITE_KV:
            self._kv_written.add(task.cell)
            self.persisted_kv.add(task.cell)
            self._commit_cell_if_ready(task.cell)
        elif task.kind == TaskKind.WRITE_H:
            self._h_written.add(task.cell)
            self.persisted_h.add(task.cell)
            if (
                task.cell.layer + 1 < self.layer_count
                and self._cell(task.cell.chunk, task.cell.layer + 1) in self.cell_done
            ):
                self.state.drop_h(task.cell.range)
                self.resident_h.discard(task.cell)
            self._commit_cell_if_ready(task.cell)

    def _commit_cell_if_ready(self, cell: Cell) -> None:
        while self.state.layer_progress[cell.layer] < self.chunk_count:
            next_cell = self._cell(self.state.layer_progress[cell.layer], cell.layer)
            if next_cell not in self._kv_written or next_cell not in self._h_written:
                return
            self.state.mark_prefill_cell_persisted(next_cell.index)

    def _run_task(self, task: Task) -> Task:
        if task.kind == TaskKind.READ_PARAMS:
            self.model.ensure_layer(task.layer)
        elif task.kind == TaskKind.COMPUTE_CELL:
            self._compute_cell(task.cell)
        elif task.kind == TaskKind.READ_KV:
            self.state.load_prefill_kv(task.cell.index)
        elif task.kind == TaskKind.WRITE_KV:
            self.state.save_prefill_kv(task.cell.index)
        elif task.kind == TaskKind.READ_H:
            self.state.read_h(task.cell.range)
        elif task.kind == TaskKind.WRITE_H:
            self.state.save_prefill_h(task.cell.index)
        return task

    def _compute_cell(self, cell: Cell) -> None:
        with torch.inference_mode():
            pos_emb = self.prefill_inputs.pos_emb.slice(cell.start, cell.end)
            if cell.layer == 0:
                hidden_states = self.prefill_inputs.inputs_embeds[:, cell.start : cell.end, :]
            else:
                hidden_states = self.state.get_h(Range(cell.layer - 1, cell.start, cell.end))

            hidden_states = self.model._prefill_forward_chunk_layer(cell.index, hidden_states, pos_emb)
            if cell.layer < len(self.prefill_inputs.deepstack_visual_embeds):
                hidden_states = self.model._deepstack_process(
                    hidden_states,
                    self.prefill_inputs.visual_pos_masks[0],
                    self.prefill_inputs.deepstack_visual_embeds[cell.layer],
                    start=cell.start,
                    end=cell.end,
                )
            self.state.update_h(cell.range, hidden_states)

    def _ready_io_tasks(self) -> list[Task]:
        tasks = []
        for layer in range(self.layer_count):
            task = Task(f"read_params:{layer}", TaskKind.READ_PARAMS, Resource.IO, layer=layer)
            if layer not in self.resident_params and task.id not in self.completed_tasks:
                tasks.append(task)

        for cell in self.cells:
            if cell in self.cell_done:
                if cell not in self.persisted_kv and cell not in self._kv_written:
                    tasks.append(self._task(TaskKind.WRITE_KV, Resource.IO, cell))
                if cell not in self.persisted_h and cell not in self._h_written:
                    tasks.append(self._task(TaskKind.WRITE_H, Resource.IO, cell))
            if cell in self.persisted_kv and cell not in self.resident_kv:
                tasks.append(self._task(TaskKind.READ_KV, Resource.IO, cell))
        for cell in self.cells:
            if cell.layer == 0 or cell in self.cell_done or cell.layer not in self.resident_params:
                continue
            previous = self._cell(cell.chunk, cell.layer - 1)
            if previous in self.persisted_h and previous not in self.resident_h:
                tasks.append(self._task(TaskKind.READ_H, Resource.IO, previous))
        return tasks

    def _ready_compute_tasks(self) -> list[Task]:
        tasks = []
        for cell in self.cells:
            if cell in self.cell_done:
                continue
            if cell.layer not in self.resident_params:
                continue
            if cell.chunk > 0 and self._cell(cell.chunk - 1, cell.layer) not in self.resident_kv:
                continue
            if cell.layer > 0 and self._cell(cell.chunk, cell.layer - 1) not in self.resident_h:
                continue
            tasks.append(self._task(TaskKind.COMPUTE_CELL, Resource.COMPUTE, cell))
        return tasks

    def _snapshot(self) -> ScheduleSnapshot:
        return ScheduleSnapshot(
            chunk_count=self.chunk_count,
            layer_count=self.layer_count,
            layer_group_size=self.model.runtime_config.prefill_schedule_layer_group_size,
            completed_tasks=frozenset(self.completed_tasks),
            cell_done=frozenset(self.cell_done),
            resident_kv=frozenset(self.resident_kv),
            resident_h=frozenset(self.resident_h),
            resident_params=frozenset(self.resident_params),
            persisted_kv=frozenset(self.persisted_kv),
            persisted_h=frozenset(self.persisted_h),
            running_io_task=self._running_io_task,
            running_compute_task=self._running_compute_task,
        )

    def _cell(self, chunk: int, layer: int) -> Cell:
        start = chunk * self.chunk_size
        return Cell(chunk, layer, start, min(start + self.chunk_size, self.prompt_len))

    def _task(self, kind: TaskKind, resource: Resource, cell: Cell) -> Task:
        return Task(self._task_id(kind, cell), kind, resource, cell=cell, layer=cell.layer)

    def _task_id(self, kind: TaskKind, cell: Cell) -> str:
        return f"{kind.value}:{cell.layer}:{cell.chunk}"
