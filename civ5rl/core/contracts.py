"""Typed contracts for the collector's public dictionary boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NotRequired, TypedDict

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


# Case identifiers and digests have two distinct scopes:
# - Provenance.case_id: manifest entry index (0..len(cases)-1) assigned by
#   case_manifest() within trusted_manifest["cases"].
# - ScenarioSpec.case_id: protocol scenario/layout identifier (may be shared
#   by a side-swapped pair in paired evaluations).
# - manifest_entry_digest(spec) (Provenance.case_hash): digest of spec
#   excluding "provenance", binding a case to its manifest entry.
# - case_content_hash(case) (ScenarioSpec.case_hash): self-digest of case
#   excluding "case_hash", verifying serialized case content.
class Provenance(TypedDict):
    source: str
    split: str
    namespace: str
    case_id: int
    case_hash: str
    scenario_hash: str
    manifest_hash: str


class BaseScenarioSpec(TypedDict):
    scenario: Scenario
    armies: list[list[str]]
    learner_side: int
    max_rounds: int
    max_calls_per_turn: int
    opponent: str
    opponent_search: NotRequired[dict[str, Any]]
    reward_profile: NotRequired[str]
    agent_seeds: list[int]
    diagnostics: NotRequired[bool]
    learner_decision_limit: NotRequired[int]
    record_step_events: NotRequired[bool]


class ProvenanceFields(TypedDict, total=False):
    episode_id: int
    case_id: int
    case_hash: str
    layout_id: int
    physical_hash: str
    streams: dict[str, int]
    parent_world: int
    placement: str
    rejected: list[dict[str, str]]
    split: str
    namespace: str
    source: str
    scenario_hash: str
    manifest_hash: str
    provenance: Provenance


class ScenarioSpec(BaseScenarioSpec, ProvenanceFields):
    diagnostic_stage: NotRequired[str]
    diagnostic_protocol: NotRequired[str]


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
    diagnostic_events: NotRequired[list[dict[str, Any]]]


class DecisionState(TypedDict):
    done: Literal[False]
    observation: dict[str, Any]
    actions: tuple[Action, ...]


class FinishedState(TypedDict):
    done: Literal[True]
    result: EpisodeResult
    joint_commands: list[dict[str, Any]]


WorkerState = DecisionState | FinishedState


class PublishedDecisionState(DecisionState):
    encoded_inline: NotRequired[Encoded]
    encoded_shape: NotRequired[tuple[int, int]]


class RepeatedDoneState(TypedDict):
    done: Literal[True]


PublishedWorkerState = PublishedDecisionState | FinishedState | RepeatedDoneState


class WorkerCommand(NamedTuple):
    slot: int
    generation: int | None
    action: Action


class WorkerStepItem(NamedTuple):
    slot: int
    generation: int
    state: PublishedWorkerState
    completed_at: float
    diagnostics: dict[str, Any] | None = None


class WorkerPayload(TypedDict):
    states: NotRequired[list[PublishedWorkerState]]
    worker_encoding_seconds: float
    wave: NotRequired[int]
    items: NotRequired[list[WorkerStepItem]]
    published_at: NotRequired[float]
    worker_request_seconds: NotRequired[float]
    worker_request_slices: NotRequired[int]
    worker_request_slice_max_seconds: NotRequired[float]


class Episode(TypedDict):
    spec: ScenarioSpec
    observations: list[dict[str, Any]]
    actions: list[tuple[Action, ...]]
    chosen: list[int]
    old_logp: list[float]
    old_log_probs: list[np.ndarray]
    old_value: list[float]
    policy_temperature: NotRequired[float]
    old_hidden: NotRequired[list[np.ndarray | None]]
    decision_metrics: list[dict[str, str | bool | float]]
    encoded: NotRequired[list[Encoded]]
    encoded_variant: NotRequired[str]
    result: NotRequired[EpisodeResult]
    joint_commands: NotRequired[list[dict[str, Any]]]
