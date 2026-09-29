"""Data structures for the idea agent.

Every structure here is serialisable so a run can be dumped to JSON and
audited later. Provenance is tracked per component: corpus-grounded evidence
carries a citation, model-supplied evidence carries an audit verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Register = Literal["corpus", "model"]
Audit = Literal["corroborated", "contested", "uncorroborated", ""]


@dataclass
class ComponentEvidence:
    """One component of a forged concept, with its provenance.

    Attributes:
        register: "corpus" for retrieved-text-grounded, "model" for LLM-supplied.
        claim: The component's contribution to the concept, in one sentence.
        chunk_id: Corpus chunk id when register is "corpus".
        article_title: Source article title when register is "corpus".
        article_url: Source article URL when register is "corpus".
        similarity: Cosine similarity of the chunk at retrieval time.
        quote: Verbatim excerpt supporting the claim.
        audit: Verdict from checking a model-register claim against the corpus.
        audit_note: Evidence for the audit verdict, with citations.
    """

    register: Register
    claim: str
    chunk_id: str = ""
    article_title: str = ""
    article_url: str = ""
    similarity: float = 0.0
    quote: str = ""
    audit: Audit = ""
    audit_note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComponentEvidence":
        register = data.get("register", "model")
        if register not in ("corpus", "model"):
            register = "model"
        audit = data.get("audit", "") or ""
        if audit not in ("corroborated", "contested", "uncorroborated"):
            audit = ""
        try:
            similarity = float(data.get("similarity", 0.0) or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        return cls(
            register=register,
            claim=str(data.get("claim", "")).strip(),
            chunk_id=str(data.get("chunk_id", "") or ""),
            article_title=str(data.get("article_title", "") or ""),
            article_url=str(data.get("article_url", "") or ""),
            similarity=similarity,
            quote=str(data.get("quote", "") or "").strip(),
            audit=audit,
            audit_note=str(data.get("audit_note", "") or "").strip(),
        )


@dataclass
class NoveltyCheck:
    """Result of embedding an idea and searching the corpus with it.

    A low max_similarity means the idea occupies space the corpus does not
    cover; a high one means it is a tracing of something already written.

    Attributes:
        max_similarity: Highest cosine similarity found against the corpus.
        nearest_title: Article title of the nearest hit.
        nearest_url: URL of the nearest hit.
        verdict: "novel", "borderline", or "tracing".
    """

    max_similarity: float = 0.0
    nearest_title: str = ""
    nearest_url: str = ""
    verdict: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoveltyCheck":
        try:
            max_similarity = float(data.get("max_similarity", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_similarity = 0.0
        return cls(
            max_similarity=max_similarity,
            nearest_title=str(data.get("nearest_title", "") or ""),
            nearest_url=str(data.get("nearest_url", "") or ""),
            verdict=str(data.get("verdict", "") or ""),
        )


@dataclass
class IdeaCard:
    """A forged concept, its composition, and its evidence trail.

    Attributes:
        name: The concept's name.
        statement: The idea in one or two sentences.
        composition: What the concept is made of and why the components fuse.
        components: Provenance-tagged evidence for each component.
        stakes: What the concept makes thinkable that was not before.
        objection: The strongest argument against it.
        novelty: Corpus-distance check result.
        walk_ids: Which walks contributed material.
        confidence: Final 0.0-1.0 score after critique and any caps.
        verdict: The referee's ruling — keep, revise, discard, or unjudged.
        tracing_of: Established concept this merely restates, if the referee
            named one. Non-empty means the card is a tracing, not a map.
        confidence_note: Why confidence was capped below the referee's own
            score. Empty when the referee's score stood unmodified.
    """

    name: str
    statement: str
    composition: str = ""
    components: list[ComponentEvidence] = field(default_factory=list)
    stakes: str = ""
    objection: str = ""
    novelty: NoveltyCheck = field(default_factory=NoveltyCheck)
    walk_ids: list[int] = field(default_factory=list)
    confidence: float = 0.0
    verdict: str = "unjudged"
    tracing_of: str = ""
    confidence_note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdeaCard":
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return cls(
            name=str(data.get("name", "")).strip() or "unnamed concept",
            statement=str(data.get("statement", "")).strip(),
            composition=str(data.get("composition", "") or "").strip(),
            components=[
                ComponentEvidence.from_dict(c) for c in data.get("components", []) or []
            ],
            stakes=str(data.get("stakes", "") or "").strip(),
            objection=str(data.get("objection", "") or "").strip(),
            novelty=NoveltyCheck.from_dict(data.get("novelty", {}) or {}),
            walk_ids=[int(w) for w in data.get("walk_ids", []) or [] if _is_int(w)],
            confidence=max(0.0, min(1.0, confidence)),
            verdict=str(data.get("verdict", "unjudged") or "unjudged"),
            tracing_of=str(data.get("tracing_of", "") or ""),
            confidence_note=str(data.get("confidence_note", "") or ""),
        )

    def corpus_components(self) -> list[ComponentEvidence]:
        return [c for c in self.components if c.register == "corpus"]

    def model_components(self) -> list[ComponentEvidence]:
        return [c for c in self.components if c.register == "model"]

    def contested(self) -> bool:
        return any(c.audit == "contested" for c in self.components)

    def is_tracing(self) -> bool:
        """Whether the referee judged this a restatement of existing work."""
        return bool(self.tracing_of.strip())


def _is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


@dataclass
class Fragment:
    """One retrieved chunk, normalised from a TraversalStep.

    Attributes:
        walk_id: Which walk produced it.
        chunk_id: Corpus chunk id.
        text: Chunk text.
        article_title: Source article title.
        article_url: Source article URL.
        similarity: Cosine similarity to the query at that step.
        depth: Step index in the walk.
        forced_jump: Whether this step was a forced global jump.
        walk_seed: The query that seeded the walk.
    """

    walk_id: int
    chunk_id: str
    text: str
    article_title: str
    article_url: str
    similarity: float
    depth: int
    forced_jump: bool
    walk_seed: str = ""


@dataclass
class WalkPlan:
    """A traversal the LLM has asked for, expressed in engine knobs.

    Attributes:
        seed: Starting concept or query phrase.
        rationale: Why this walk should be run.
        depth: Step budget.
        epsilon: Exploration probability.
        temperature: Softmax temperature for the exploit path.
        top_k: Candidates considered per step.
    """

    seed: str
    rationale: str = ""
    depth: int = 8
    epsilon: float = 0.1
    temperature: float = 1.0
    top_k: int = 20

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WalkPlan":
        seed = str(data.get("seed", "") or "").strip()
        if not seed:
            raise ValueError("walk plan is missing a seed")
        return cls(
            seed=seed,
            rationale=str(data.get("rationale", "") or "").strip(),
            depth=_clamped_int(data.get("depth"), 2, 24, 8),
            epsilon=_clamped_float(data.get("epsilon"), 0.0, 1.0, 0.1),
            temperature=_clamped_float(data.get("temperature"), 0.0, 4.0, 1.0),
            top_k=_clamped_int(data.get("top_k"), 3, 60, 20),
        )


@dataclass
class WalkResult:
    """Outcome of one traversal.

    Attributes:
        walk_id: Sequence number within the run.
        plan: The plan that produced it.
        fragments: Chunks collected along the path.
        jumps: Number of forced global jumps taken.
        note: The LLM's assessment of what this walk yielded.
    """

    walk_id: int
    plan: WalkPlan
    fragments: list[Fragment] = field(default_factory=list)
    jumps: int = 0
    note: str = ""

    def to_summary(self) -> dict[str, Any]:
        return {
            "walk_id": self.walk_id,
            "seed": self.plan.seed,
            "depth": len(self.fragments),
            "jumps": self.jumps,
            "articles": len({f.article_title for f in self.fragments}),
        }


@dataclass
class IdeaRun:
    """A complete agent run: seed, walks, cards, and the final report.

    Attributes:
        seed: The user's starting concept or prompt.
        brief: The LLM's unpacking of the seed (sub-tensions, framing).
        walks: Every traversal performed.
        cards: Forged ideas, in the order produced.
        rounds: Number of synthesize/refine rounds executed.
        summary: Closing assessment written by the LLM.
    """

    seed: str
    brief: str = ""
    walks: list[WalkResult] = field(default_factory=list)
    cards: list[IdeaCard] = field(default_factory=list)
    rounds: int = 0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "brief": self.brief,
            "rounds": self.rounds,
            "summary": self.summary,
            "walks": [
                {
                    "walk_id": w.walk_id,
                    "plan": asdict(w.plan),
                    "jumps": w.jumps,
                    "note": w.note,
                    "fragments": [asdict(f) for f in w.fragments],
                }
                for w in self.walks
            ],
            "cards": [asdict(c) for c in self.cards],
        }

    def kept_cards(self, threshold: float = 0.5) -> list[IdeaCard]:
        return [c for c in self.cards if c.confidence >= threshold]


def _clamped_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _clamped_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default
