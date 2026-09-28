"""The Demerzel engine: the minimal core loop (V1.md §4.2).

Wired, not invented: the engine composes the mechanisms of Phases 0–6 behind
their interfaces and runs the closed loop::

    WRITE -> INDEX -> RETRIEVE -> USE -> EVALUATE -> UPDATE

Responsibilities
----------------
* Build the layers from a :class:`~demerzel.config.Config` and the reference
  implementations (or injected seams, for tests).
* Run an end-to-end query: remote route -> retrieve -> sufficiency ->
  build context -> answer -> MPE step -> pointer updates.
* Emit the standard trace (§5.5) and log a reproducibility block (§14).

Ablations
---------
Every mechanism carries a config switch; ``Engine`` honours them so the
Experiment-1/2/4 controls are config-only::

    config.ablations           # {'decay': bool, 'mpe': bool, 'graph': bool}
    config.mpe.enabled
    config.decay.enabled
    config.retrieval.graph_enabled

Experiment 2 invariant — G fixed, E fixed, A mutable — is enforced by the
:class:`~demerzel.mpe.updater.MPEFeedback` design (it never writes the graph),
which the P0-5 test asserts directly on the engine.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from .assemblage.answerer import Answerer
from .assemblage.context_builder import ContextBuilder
from .assemblage.query_router import QueryRouter, RouteDecision
from .assemblage.reranker import Reranker
from .assemblage.sufficiency_router import SufficiencyRouter
from .config import Config
from .core.context_distance import ContextDistance
from .core.pointer_decay import PointerDecay
from .entropy.context_decay import ContextDecay
from .entropy.forgetting_log import ForgettingLog
from .foundation.ingestion import IngestionPipeline
from .foundation.raw_store import RawStore
from .halo.context_state import ContextState
from .halo.ring_buffer import RingBuffer
from .interfaces.graph import GraphBackend
from .interfaces.pointer import PointerUpdater
from .interfaces.providers import EmbeddingProvider, LLMProvider
from .interfaces.storage import MemoryRecord, StorageBackend
from .mpe.updater import MPEFeedback
from .nexus.nodes import NexusBuilder, node_id_for_memory
from .nexus.pointer_update import NexusPointerWriteback
from .nexus.ppr import PPRRetriever
from .nexus.retrieval import LexicalRetriever, VectorRetriever
from .nexus.scoring import Candidate, HybridScorer, RetrievalResult
from .trace import RunTrace


class Engine:
    """Compose the layers and run the core loop."""

    def __init__(
        self,
        storage: StorageBackend,
        graph: GraphBackend,
        embedder: EmbeddingProvider,
        evidence_store,
        llm: Optional[LLMProvider] = None,
        config: Optional[Config] = None,
        raw_store: Optional[RawStore] = None,
        run_trace: Optional[RunTrace] = None,
        forgetting_log: Optional[ForgettingLog] = None,
    ):
        self.config = (config or Config()).apply_ablations()
        self.storage = storage
        self.graph = graph
        self.embedder = embedder
        self.llm = llm
        self._raw = raw_store or RawStore(evidence_store)

        # Halo + context state.
        self.ring = RingBuffer(self.config.halo.max_turns)
        self.context_state = ContextState(
            embedder, self.config.context_distance.alpha, self.config.context_distance.beta
        )

        # Core mechanisms.
        self.context_distance = ContextDistance(
            embedder,
            cold_start_k=self.config.context_distance.cold_start_k,
            cold_start_enabled=self.config.context_distance.cold_start_enabled,
        )
        self.pointer_decay = PointerDecay(
            storage,
            scale=self.config.decay.scale,
            mode=self.config.decay.mode,
            enabled=self.config.decay.enabled,
        )
        self.mpe = MPEFeedback(
            updater=self._make_updater(),
            enabled=self.config.mpe.enabled,
        )

        # Nexus layer.
        self.builder = NexusBuilder(graph, enabled=self.config.retrieval.graph_enabled)
        self.ppr = PPRRetriever(
            graph, self.config.retrieval.ppr_damping, self.config.retrieval.graph_enabled
        )
        self.lexical = LexicalRetriever(enabled=self.config.retrieval.lexical_enabled)
        self.vector = VectorRetriever(embedder, enabled=self.config.retrieval.vector_enabled)
        self.scorer = HybridScorer(
            self.config.retrieval.w_semantic,
            self.config.retrieval.w_ppr,
            self.config.retrieval.w_pointer,
            self.config.retrieval.w_context_distance,
            self.config.retrieval.w_lexical,
            self.config.retrieval.w_vector,
        )
        self.writeback = NexusPointerWriteback(graph)

        # Entropy / forgetting.
        self.forgetting_log = forgetting_log or ForgettingLog()
        self.context_decay = ContextDecay(
            storage,
            self.context_distance,
            self.pointer_decay,
            self.forgetting_log,
            enabled=self.config.decay.enabled,
        )

        # Assemblage.
        self.router = QueryRouter()
        self.sufficiency = SufficiencyRouter(
            self.config.sufficiency.threshold, self.config.sufficiency.max_fallback_depth
        )
        self.context_builder = ContextBuilder(self.config.retrieval.top_k)
        self.answerer = Answerer(llm, mode="extractive")
        self.reranker = Reranker(enabled=False)

        # Ingestion.
        self.ingestion = IngestionPipeline(
            self._raw, storage, default_memory_type="fact"
        )

        self.run_trace = run_trace

    def _make_updater(self) -> PointerUpdater:
        from .interfaces.pointer import EtaPointerUpdater

        return EtaPointerUpdater(
            self.storage, eta=self.config.mpe.eta, enabled=self.config.mpe.enabled
        )

    # -- ingestion (WRITE / INDEX) ------------------------------------------
    def ingest(
        self,
        session_id: str,
        text: str,
        speaker: str = "user",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Ingest one turn; returns the Chronicle memory id."""

        result = self.ingestion.ingest_turn(
            session_id=session_id, text=text, speaker=speaker, metadata=metadata
        )
        # Halo holds the shallow copy too; a bound ring auto-evicts.
        self.ring.append(text, speaker=speaker)

        # Index graph nodes/edges for the new record.
        record = self.storage.get(result.memory_ids[0])
        if record is not None:
            self.builder.add_memory_node(record)
            self.builder.link_episode(record)

        if self.run_trace is not None:
            self.run_trace.log_event(
                {"event": "ingest", "turn_id": result.turn_id, "memory_id": result.memory_ids[0]}
            )
        return result.memory_ids[0]

    def ingest_batch(self, session_id: str, turns: Sequence[Dict[str, Any]]) -> List[str]:
        """Ingest many turns (each a ``{'text':..,'speaker':..}`` dict)."""
        out = []
        for turn in turns:
            out.append(
                self.ingest(
                    session_id,
                    turn["text"],
                    turn.get("speaker", "user"),
                    turn.get("metadata"),
                )
            )
        return out

    # -- retrieval (RETRIEVE) -------------------------------------------------
    def _current_state_vector(self) -> List[float]:
        conversation = [t.text for t in self.ring.all()]
        return self.context_state.state_vector(conversation, conversation)

    def retrieve(
        self, query: str, state_vector: Optional[List[float]] = None
    ) -> RetrievalResult:
        """Return the scored candidates for ``query``."""
        records = [r for r in self.storage.list_all() if not r.retired]
        sv = state_vector if state_vector is not None else self._current_state_vector()

        # Per-record context distance (cold start resolves to 0).
        distances: Dict[str, float] = {}
        for r in records:
            distances[r.id] = self.context_distance.compute(
                r.content, sv, len(self.ring.all())
            ).distance

        lexical = self.lexical.score(query, records)
        vector = self.vector.score(query, records)
        ppr = {}
        seed_nodes = [node_id_for_memory(r.id) for r in records]
        if self.config.retrieval.graph_enabled and seed_nodes:
            ppr = self.ppr.memory_scores(seed_nodes)

        candidates = self.scorer.score(
            records, lexical, vector, ppr, distances, backend="hybrid"
        )
        return RetrievalResult(
            query=query,
            candidates=candidates,
            backend="hybrid",
            depth=1,
            seed_node_ids=seed_nodes,
        )

    # -- end-to-end query (USE/EVALUATE/UPDATE) ------------------------------
    def query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Run the full loop and return a JSON-serialisable trace record."""
        start = time.time()

        # 1. route
        route: RouteDecision = self.router.route(query)

        # 2. retrieve
        result = self.retrieve(query)

        # 3. sufficiency
        decision = self.sufficiency.assess(result)

        # 4. build context + answer
        records_by_id = {r.id: r for r in self.storage.list_all()}
        built = self.context_builder.build(query, result.candidates, records_by_id)
        answer = self.answerer.answer(query, built.passage, built.used_evidence)

        # 5. decay cycle (context-distance drift) — mutates pointers
        sv = self._current_state_vector()
        turn_count = len(self.ring.all())
        decay_outcome = self.context_decay.run(
            [r for r in records_by_id.values() if not r.retired], sv, turn_count
        )

        # 6. MPE step on the top candidate used by the answer
        used = result.ids()[:1]
        pointer_updates: List[dict] = []
        mpe_value: Optional[float] = None
        if used:
            memory_id = used[0]
            memory = self.storage.get(memory_id)
            if memory is not None:
                # U_predicted from pre-retrieval channels only.
                sim = self.vector.similarity(query, memory) if self.vector.enabled else 0.0
                predicted_ev = context_score = 0.0
                # evidence score = similarity actually used
                evidence = sim
                outcome = self.mpe.step(
                    memory_id=memory_id,
                    similarity=sim,
                    pointer=float(memory.pointer_strength),
                    context_distance=dist_for(memory_id, decay_outcome.distances),
                    evidence_score=evidence,
                )
                if outcome is not None:
                    mpe_value = outcome.clipped
                    new_point = self.storage.get(memory_id).pointer_strength
                    pointer_updates.append(
                        {
                            "memory_id": memory_id,
                            "old": round(memory.pointer_strength, 6),
                            "new": round(new_point, 6),
                            "reason": "positive_mpe" if outcome.positive else "negative_mpe",
                            "mpe": round(outcome.clipped, 6),
                        }
                    )
                    if self.run_trace is not None:
                        self.run_trace.log_pointer_update(pointer_updates[-1])

        elapsed = time.time() - start
        record = {
            "run_id": self.config.run_id,
            "query": query,
            "candidates": [c.as_trace_dict() for c in result.candidates],
            "used_evidence": built.used_evidence,
            "mpe": mpe_value,
            "pointer_updates": pointer_updates,
            "metrics": {
                "elapsed_s": round(elapsed, 4),
                "candidate_count": len(result.candidates),
                "token_count": built.token_count,
                "route": route.backend,
                "sufficient": decision.sufficient,
                "fallback_layer": decision.next_layer,
            },
        }
        if self.run_trace is not None:
            self.run_trace.log_retrieval(record, query)
            self.run_trace.log_answer(answer.as_dict())
        return record

    @property
    def memory_count(self) -> int:
        return len(self.storage.list_all())

    @property
    def evidence(self) -> RawStore:
        """The append-only Foundation store (read-only access for harnesses)."""
        return self._raw


def dist_for(memory_id: str, distances: Dict[str, float]) -> float:
    return float(distances.get(memory_id, 0.0))


def build_default_engine(
    config: Optional[Config] = None,
    storage=None,
    graph=None,
    embedder=None,
    evidence_store=None,
    llm=None,
    run_trace: Optional[RunTrace] = None,
) -> Engine:
    """Wire the stock reference implementations behind the interfaces.

    This is the one place the reference layer is named. Core mechanisms and the
    engine itself never import it, so swapping SQLite for memory storage (or the
    hashed embedder for a real one) changes nothing above this line.
    """
    from .reference import (
        DictGraph,
        HashingEmbeddingProvider,
        InMemoryEvidenceStore,
        InMemoryStorage,
        MockLLMProvider,
    )

    cfg = (config or Config()).apply_ablations()
    storage = storage if storage is not None else InMemoryStorage()
    graph = graph if graph is not None else DictGraph()
    embedder = embedder if embedder is not None else HashingEmbeddingProvider()
    evidence_store = (
        evidence_store if evidence_store is not None else InMemoryEvidenceStore()
    )
    llm = llm if llm is not None else MockLLMProvider()
    return Engine(
        storage,
        graph,
        embedder,
        evidence_store,
        llm,
        cfg,
        run_trace=run_trace,
    )


__all__ = ["Engine", "build_default_engine", "dist_for"]