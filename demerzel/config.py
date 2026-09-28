"""Configuration system.

Every mechanism weight and every ablation switch named in V1.md is a
config field, so an experiment is a config file rather than a code change
(V1.md §7, §11.2). The loader is deliberately dependency-free: YAML when
PyYAML happens to be installed, JSON always.

Config hash (V1.md §14) is derived from the *canonical* JSON encoding of the
whole config, so two runs that differ in any recorded knob cannot collide.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class DecayConfig:
    """Context-distance pointer decay (V1.md §7.2)."""

    enabled: bool = True
    #: Scale S in ``R = exp(-d / S)``. d=0 gives R=1 (no decay).
    scale: float = 1.0
    #: Experiment 1 selector: none | time | boundary | context | context_mpe
    mode: str = "context"
    #: Ablation C (HingeMem-style) threshold for the discrete boundary mode.
    boundary_threshold: float = 0.5
    #: Ablation B (Ebbinghaus) half-life in turns.
    time_half_life_turns: float = 50.0


@dataclass
class MPEConfig:
    """MPE feedback loop (V1.md §7.3, §7.4)."""

    enabled: bool = True
    #: Learning rate eta in ``P_next = clip(P + eta*MPE, 0, 1)``.
    eta: float = 0.05
    #: U_predicted blend: predicted utility is an affordance estimate only.
    pred_w_similarity: float = 0.5
    pred_w_pointer: float = 0.3
    pred_w_proximity: float = 0.2
    #: U_actual weights (§7.3 first-stage defaults: evidence only).
    evidence_weight: float = 1.0
    answer_delta_weight: float = 0.0
    retrieval_delta_weight: float = 0.0
    token_cost_weight: float = 0.0
    #: Bound applied to MPE before it reaches the updater.
    mpe_clip: float = 1.0


@dataclass
class RetrievalConfig:
    """Hybrid retrieval scoring (V1.md §7.5) and backend switches.

    The accessibility prior ``w_pointer`` is deliberately not tiny relative to
    the semantic weight: the central Exp-2 claim requires that moving ``A`` can
    actually reorder retrieval, and a small pointer weight lets a strong
    semantic match swamp it (see ``docs/test_report.md`` §"Exp-2 prerequisite").
    All channels are normalised to ``[0, 1]`` before weighting, so the weights
    are comparable magnitudes.
    """

    w_semantic: float = 1.0
    w_ppr: float = 0.6
    w_pointer: float = 1.0
    w_context_distance: float = 0.4
    #: Ablation switch for Nexus graph retrieval (P1: graph.enabled).
    graph_enabled: bool = True
    lexical_enabled: bool = True
    vector_enabled: bool = True
    ppr_damping: float = 0.85
    top_k: int = 5
    #: Hybrid score blends lexical/vector semantic channels.
    w_lexical: float = 0.5
    w_vector: float = 0.5


@dataclass
class ContextDistanceConfig:
    """Context distance (V1.md §7.1) and its cold-start rule (§9.1)."""

    #: state = alpha * conversation_embedding + beta * halo_centroid
    alpha: float = 0.5
    beta: float = 0.5
    #: K in the cold-start rule: turns below this force d = 0.
    cold_start_k: int = 3
    #: When true, the cold-start rule applies at all (Invariant 7 toggle).
    cold_start_enabled: bool = True


@dataclass
class UtilityConfig:
    """U_actual weights, mirroring V1.md §7.3's YAML block."""

    evidence_weight: float = 1.0
    answer_delta_weight: float = 0.0
    retrieval_delta_weight: float = 0.0
    token_cost_weight: float = 0.0


@dataclass
class HaloConfig:
    """Shallow-memory ring buffer (V1.md §5.4)."""

    max_turns: int = 8
    #: eviction policy: evict | merge | retain
    policy: str = "evict"


@dataclass
class SufficiencyConfig:
    """Sufficiency routing (V1.md §7.7)."""

    #: Minimum hybrid score for Nexus/Halo to be called sufficient.
    threshold: float = 0.35
    #: Maximum hops the router will fall back through.
    max_fallback_depth: int = 2


@dataclass
class StorageConfig:
    """Storage selection. ``path`` of ``:memory:`` means in-memory SQLite."""

    backend: str = "sqlite"
    path: str = "demerzel.db"
    #: fsync on every write; the crash-safety knob behind P0-7.
    synchronous: str = "FULL"


@dataclass
class TraceConfig:
    """Standard trace output (V1.md §5.5, §14)."""

    enabled: bool = True
    output_dir: str = "results"


@dataclass
class Config:
    """The complete, hashable run configuration."""

    run_id: str = "run_local"
    seed: int = 0
    system_version: str = "v0.1.0"

    storage: StorageConfig = field(default_factory=StorageConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    mpe: MPEConfig = field(default_factory=MPEConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    context_distance: ContextDistanceConfig = field(default_factory=ContextDistanceConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    halo: HaloConfig = field(default_factory=HaloConfig)
    sufficiency: SufficiencyConfig = field(default_factory=SufficiencyConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)

    providers: Dict[str, Any] = field(
        default_factory=lambda: {
            "llm": "mock",
            "embedding": "mock",
            "model": "mock-deterministic",
            "temperature": 0.0,
        }
    )
    dataset_version: str = "synthetic-v1"

    #: Ablation entry point required by P1: one switch per mechanism.
    ablations: Dict[str, bool] = field(
        default_factory=lambda: {"decay": True, "mpe": True, "graph": True}
    )

    def apply_ablations(self) -> "Config":
        """Return a copy with the ablation switches applied to mechanisms.

        This is the one place the switches named in P1 map onto the mechanism
        configs, so an experiment cannot silently disagree with its label.
        Each switch is ANDed with the mechanism's own ``enabled`` flag, so a
        mechanism switched off directly stays off.
        """
        cfg = replace(
            self,
            decay=replace(
                self.decay,
                enabled=self.decay.enabled and self.ablations.get("decay", True),
            ),
            mpe=replace(
                self.mpe,
                enabled=self.mpe.enabled and self.ablations.get("mpe", True),
            ),
            retrieval=replace(
                self.retrieval,
                graph_enabled=(
                    self.retrieval.graph_enabled
                    and self.ablations.get("graph", True)
                ),
            ),
        )
        return cfg

    def canonical(self) -> Dict[str, Any]:
        """Deterministic, JSON-safe view used for hashing and metadata."""
        return asdict(self)

    def config_hash(self) -> str:
        """Stable digest over the whole config (V1.md §14)."""
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return self.canonical()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Rebuild a Config from a mapping, ignoring unknown keys."""
        data = dict(data or {})
        kwargs: Dict[str, Any] = {}
        for name, ftype in (
            ("storage", StorageConfig),
            ("decay", DecayConfig),
            ("mpe", MPEConfig),
            ("retrieval", RetrievalConfig),
            ("context_distance", ContextDistanceConfig),
            ("utility", UtilityConfig),
            ("halo", HaloConfig),
            ("sufficiency", SufficiencyConfig),
            ("trace", TraceConfig),
        ):
            if name in data and isinstance(data[name], Mapping):
                known = {f.name for f in ftype.__dataclass_fields__.values()}
                kwargs[name] = ftype(
                    **{k: v for k, v in data[name].items() if k in known}
                )
        for name in (
            "run_id",
            "seed",
            "system_version",
            "providers",
            "dataset_version",
            "ablations",
        ):
            if name in data:
                kwargs[name] = data[name]
        return cls(**kwargs)


def load_config(path: Optional[str] = None) -> Config:
    """Load a config from ``path``, or return defaults when omitted.

    YAML is used when PyYAML is importable; JSON always works. A missing file
    is an error rather than a silent default, because silently running the
    wrong config is exactly the reproducibility failure §14 guards against.
    """
    if path is None:
        return Config()

    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    data: Any
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PyYAML is required to load YAML configs; "
                "use JSON configs or install PyYAML"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    return Config.from_dict(data or {})


def save_config(config: Config, path: str) -> None:
    """Write the config beside the run's results (V1.md §14)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config.canonical(), handle, indent=2, sort_keys=True)


__all__ = [
    "Config",
    "ContextDistanceConfig",
    "DecayConfig",
    "HaloConfig",
    "MPEConfig",
    "RetrievalConfig",
    "StorageConfig",
    "SufficiencyConfig",
    "TraceConfig",
    "UtilityConfig",
    "load_config",
    "save_config",
]
