"""Typed contracts for the collector's public dictionary boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

if TYPE_CHECKING:
    import numpy as np

    from civ5rl.core.observation import Encoded
    from civ5sim.engine import Action


class Scenario(TypedDict):
    schema: NotRequired[int]
    kind: NotRequired[str]
    active_side: NotRequired[int]
    map: dict[str, Any]
    units: list[dict[str, Any]]
    random_damage: bool
    fog: bool
    seed: NotRequired[int]


class Provenance(TypedDict):
    source: str
    split: str
    namespace: str
    case_id: int
    case_hash: str
    scenario_hash: str
    manifest_hash: str


class ScenarioSpec(TypedDict):
    scenario: Scenario
    armies: list[list[str]]
    learner_side: int
    max_rounds: int
    max_calls_per_turn: int
    opponent: str
    agent_seeds: list[int]
    episode_id: NotRequired[int]
    case_id: NotRequired[int]
    layout_id: NotRequired[int]
    diagnostic_stage: NotRequired[str]
    physical_hash: NotRequired[str]
    streams: NotRequired[dict[str, int]]
    parent_world: NotRequired[int]
    placement: NotRequired[str]
    rejected: NotRequired[list[dict[str, str]]]
    split: NotRequired[str]
    namespace: NotRequired[str]
    source: NotRequired[str]
    scenario_hash: NotRequired[str]
    manifest_hash: NotRequired[str]
    provenance: NotRequired[Provenance]
    diagnostics: NotRequired[bool]
    diagnostic_protocol: NotRequired[str]
    learner_decision_limit: NotRequired[int]


class AgentContext(TypedDict):
    map: dict[str, Any]
    side: int
    armies: list[list[str]]
    max_rounds: int
    random_damage: bool
    fog: bool
    seed: int


class EpisodeResult(TypedDict):
    winner: int | None
    reason: str
    reward: int
    commands: int
    learner_decisions: int
    rounds: float
    timings: dict[str, float]
    first_contact_turn: int | None
    remaining_hp: list[int]
    damage_dealt: list[int]
    units_lost: list[int]


class WorkerState(TypedDict):
    done: bool
    observation: NotRequired[dict[str, Any]]
    actions: NotRequired[tuple[Action, ...]]
    result: NotRequired[EpisodeResult]
    joint_commands: NotRequired[list[dict[str, Any]]]
    encoded_inline: NotRequired[Encoded]
    encoded_shape: NotRequired[tuple[int, int]]


class WorkerPayload(TypedDict):
    states: list[WorkerState]
    worker_encoding_seconds: float


class Episode(TypedDict):
    spec: ScenarioSpec
    observations: list[dict[str, Any]]
    actions: list[tuple[Action, ...]]
    chosen: list[int]
    old_logp: list[float]
    old_log_probs: list[np.ndarray]
    old_value: list[float]
    old_hidden: list[np.ndarray | None]
    decision_metrics: list[dict[str, str | bool | float]]
    encoded: NotRequired[list[Encoded]]
    encoded_variant: NotRequired[str]
    result: NotRequired[EpisodeResult]
    joint_commands: NotRequired[list[dict[str, Any]]]
