"""The idea agent loop.

The traversal engine is used as a tool: it runs autonomously and returns loose
fragments. The LLM sits outside it — planning walks, diagnosing what came back,
re-running with different knobs, then forging and vetting concepts.

Loop:
    plan walks -> walk -> assess -> (refine and walk again) -> synthesize
    -> novelty check -> audit model claims -> critique -> summarize
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from rhizome.embedder.base import Embedder, EmbeddingError
from rhizome.ideas import prompts
from rhizome.ideas.models import (
    ComponentEvidence,
    Fragment,
    IdeaCard,
    IdeaRun,
    NoveltyCheck,
    WalkPlan,
    WalkResult,
)
from rhizome.llm.base import LLMClient, LLMError
from rhizome.traversal.config import TraversalConfig
from rhizome.traversal.engine import TraversalEngine
from rhizome.vectorstore.client import VectorStoreClient

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str, str], None]


class IdeaAgentError(Exception):
    """Raised when the agent cannot complete a run."""


def _noop_progress(stage: str, message: str) -> None:
    return None


class IdeaAgent:
    """LLM-driven idea generation over rhizomatic traversal.

    Args:
        llm: Chat-completion client.
        embedder: Embedder for novelty checks and claim audits.
        vector_store: Qdrant client pointed at the corpus collection.
        collection_name: Corpus collection to traverse and search.
        max_walks: Hard cap on traversals per run.
        walk_depth: Default step budget per walk; the planner may deviate.
        max_rounds: Synthesize/critique rounds before the run must close.
        max_ideas: Concepts requested per synthesis round.
        synthesis_model / planner_model: Model names for those roles when the
            caller wants a cheaper model on some stages. Defaults to the
            supplied client for every stage.
        novelty_tracing_threshold: Max cosine similarity at or above which an
            idea counts as a tracing of existing corpus text.
        novelty_borderline_threshold: Similarity at or above which an idea
            counts as borderline rather than novel.
        audit_top_k: Passages retrieved per model-register claim during audit.
        on_progress: Optional callback invoked as (stage, message).
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: Embedder,
        vector_store: VectorStoreClient,
        collection_name: str,
        *,
        max_walks: int = 3,
        walk_depth: int = 8,
        max_rounds: int = 2,
        max_ideas: int = 3,
        novelty_tracing_threshold: float = 0.72,
        novelty_borderline_threshold: float = 0.60,
        audit_top_k: int = 4,
        on_progress: ProgressFn | None = None,
    ):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.max_walks = max_walks
        self.walk_depth = walk_depth
        self.max_rounds = max(1, max_rounds)
        self.max_ideas = max_ideas
        self.novelty_tracing_threshold = novelty_tracing_threshold
        self.novelty_borderline_threshold = novelty_borderline_threshold
        self.audit_top_k = audit_top_k
        self.on_progress = on_progress or _noop_progress

    def run(self, seed: str) -> IdeaRun:
        """Execute a full idea-generation run.

        Args:
            seed: The user's starting concept, tension, or half-formed thought.

        Returns:
            An IdeaRun carrying every walk, every forged card, and the closing
            assessment.

        Raises:
            IdeaAgentError: If planning fails outright or the corpus is
                unreachable. Individual stage failures are logged and skipped
                so a partial run still returns usable material.
        """
        seed = (seed or "").strip()
        if not seed:
            raise IdeaAgentError("seed concept must not be empty")

        run = IdeaRun(seed=seed)
        plans = self._plan_initial(run)
        if not plans:
            raise IdeaAgentError("planner produced no walks; nothing to traverse")

        for plan in plans:
            if len(run.walks) >= self.max_walks:
                break
            self._execute_walk(run, plan)

        walk_budget = self.max_walks - len(run.walks)
        while walk_budget > 0:
            latest_note = run.walks[-1].note if run.walks else ""
            plans = self._plan_refinement(run, walk_budget, latest_note)
            if not plans:
                self.on_progress("plan", "planner declined further walks; moving to synthesis")
                break
            for plan in plans:
                if len(run.walks) >= self.max_walks:
                    break
                self._execute_walk(run, plan)
            walk_budget = self.max_walks - len(run.walks)
            if not run.walks:
                break

        if not any(w.fragments for w in run.walks):
            raise IdeaAgentError(
                "no fragments were retrieved from any walk; check corpus and embedder"
            )

        for round_index in range(self.max_rounds):
            run.rounds = round_index + 1

            if round_index > 0:
                survivor = _best_survivor(run)
                if survivor is None:
                    self.on_progress("round", "no survivor worth re-seeding from; closing run")
                    break
                if len(run.walks) >= self.max_walks:
                    self.on_progress("round", "walk budget exhausted; re-synthesizing from existing fragments")
                else:
                    self.on_progress("round", f"round {run.rounds}: re-seeding from {survivor.name}")
                    self._execute_walk(
                        run,
                        WalkPlan(
                            seed=survivor.name,
                            rationale=f"re-seed from surviving concept: {survivor.statement[:200]}",
                            depth=self.walk_depth,
                            epsilon=min(1.0, 0.25),
                            temperature=1.2,
                        ),
                    )

            new_cards = self._synthesize(run, round_index)
            if not new_cards:
                self.on_progress("synthesize", f"round {run.rounds} produced no concepts")
                continue
            self._check_novelty(new_cards)
            self._audit_model_claims(new_cards)
            self._critique(new_cards)
            run.cards.extend(new_cards)
            self.on_progress(
                "synthesize",
                f"round {run.rounds} complete: {len(new_cards)} concept(s), "
                f"{len(run.kept_cards())} kept so far",
            )

        run.summary = self._summarize(run)
        return run

    def _plan_initial(self, run: IdeaRun) -> list[WalkPlan]:
        self.on_progress("plan", f"unpacking seed: {run.seed[:80]}")
        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_PLANNER},
                    {
                        "role": "user",
                        "content": prompts.user_planner_initial(
                            run.seed, self.max_walks, self.walk_depth
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=2048,
            )
        except LLMError as exc:
            raise IdeaAgentError(f"planning failed: {exc}") from exc

        run.brief = str((data or {}).get("brief", "")).strip()
        plans = _parse_plans(data, fallback_seed=run.seed)
        if not plans:
            logger.warning("planner returned no valid walks; falling back to seed walk")
            plans = [WalkPlan(seed=run.seed, rationale="planner returned nothing usable")]
        self.on_progress("plan", f"brief: {run.brief[:120]}")
        for plan in plans:
            self.on_progress("plan", f"walk planned: {plan.seed[:70]} (eps={plan.epsilon}, temp={plan.temperature})")
        return plans[: self.max_walks]

    def _plan_refinement(self, run: IdeaRun, budget: int, latest_note: str) -> list[WalkPlan]:
        self.on_progress("plan", f"refining with budget for {budget} more walk(s)")
        summaries = [w.to_summary() for w in run.walks]
        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_PLANNER},
                    {
                        "role": "user",
                        "content": prompts.user_planner_refine(
                            run.brief, summaries, latest_note, budget, self.walk_depth
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=2048,
            )
        except LLMError as exc:
            logger.warning("refinement planning failed: %s", exc)
            return []
        plans = _parse_plans(data, fallback_seed="")
        for plan in plans:
            self.on_progress("plan", f"refined walk: {plan.seed[:70]} (eps={plan.epsilon}, temp={plan.temperature})")
        return plans[:budget]

    def _execute_walk(self, run: IdeaRun, plan: WalkPlan) -> None:
        walk_id = len(run.walks) + 1
        self.on_progress(
            "walk",
            f"[{walk_id}] traversing: {plan.seed[:70]} (depth={plan.depth})",
        )
        config = TraversalConfig(
            depth=plan.depth,
            epsilon=plan.epsilon,
            top_k=plan.top_k,
            collection_name=self.collection_name,
            temperature=plan.temperature,
        )
        engine = TraversalEngine(
            embedder=self.embedder,
            vector_store=self.vector_store,
            config=config,
        )
        try:
            steps = engine.traverse(plan.seed)
        except Exception as exc:  # noqa: BLE001 - engine raises several types
            logger.warning("walk %d failed on seed %r: %s", walk_id, plan.seed, exc)
            self.on_progress("walk", f"[{walk_id}] failed: {exc}")
            run.walks.append(WalkResult(walk_id=walk_id, plan=plan, note=f"walk failed: {exc}"))
            return

        fragments = [
            Fragment(
                walk_id=walk_id,
                chunk_id=step.chunk_id,
                text=step.text,
                article_title=step.article_title,
                article_url=step.article_url,
                similarity=step.similarity,
                depth=step.depth,
                forced_jump=step.forced_jump,
                walk_seed=plan.seed,
            )
            for step in steps
        ]
        jumps = sum(1 for f in fragments if f.forced_jump)
        result = WalkResult(walk_id=walk_id, plan=plan, fragments=fragments, jumps=jumps)
        run.walks.append(result)
        self.on_progress(
            "walk",
            f"[{walk_id}] {len(fragments)} fragments, {jumps} forced jump(s), "
            f"{len({f.article_title for f in fragments})} articles",
        )

        result.note = self._assess_walk(result, run.brief)

    def _assess_walk(self, result: WalkResult, brief: str) -> str:
        payload = [_fragment_payload(f) for f in result.fragments]
        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_PLANNER},
                    {
                        "role": "user",
                        "content": (
                            f"Run brief:\n{brief}\n\n"
                            + prompts.user_walk_assess(result.plan.seed, result.plan.rationale, payload)
                        ),
                    },
                ],
                temperature=0.4,
                max_tokens=1024,
            )
        except LLMError as exc:
            logger.warning("walk assessment failed: %s", exc)
            return f"assessment unavailable: {exc}"
        note = str((data or {}).get("note", "")).strip()
        self.on_progress("walk", f"[{result.walk_id}] note: {note[:140]}")
        return note

    def _synthesize(self, run: IdeaRun, round_index: int) -> list[IdeaCard]:
        self.on_progress("synthesize", f"round {round_index + 1}: forging concepts")
        seen: dict[str, Fragment] = {}
        for walk in run.walks:
            for fragment in walk.fragments:
                seen.setdefault(fragment.chunk_id, fragment)
        payload = [_fragment_payload(f) for f in seen.values()]

        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_SYNTHESIZER},
                    {
                        "role": "user",
                        "content": prompts.user_synthesize(
                            run.seed, run.brief, payload, self.max_ideas
                        ),
                    },
                ],
                temperature=0.9,
                max_tokens=8192,
                retries=2,
            )
        except LLMError as exc:
            logger.warning("synthesis failed: %s", exc)
            self.on_progress("synthesize", f"synthesis failed: {exc}")
            return []

        raw_ideas = (data or {}).get("ideas", []) or []
        if isinstance(raw_ideas, dict):
            raw_ideas = [raw_ideas]

        cards: list[IdeaCard] = []
        for raw in raw_ideas:
            if not isinstance(raw, dict):
                continue
            card = IdeaCard.from_dict(raw)
            if not card.statement:
                continue
            card.components = _validate_components(card.components, seen)
            cards.append(card)
            self.on_progress("synthesize", f"  concept: {card.name}")
        return cards

    def _check_novelty(self, cards: list[IdeaCard]) -> None:
        self.on_progress("verify", f"novelty-checking {len(cards)} concept(s) against the corpus")
        statements = [f"{c.name}: {c.statement}" for c in cards]
        try:
            vectors = self.embedder.embed(statements)
        except EmbeddingError as exc:
            logger.warning("novelty embedding failed: %s", exc)
            self.on_progress("verify", f"novelty check skipped: {exc}")
            return

        for card, vector in zip(cards, vectors):
            try:
                hits = self.vector_store.search(
                    query_vector=vector, top_k=3, with_vector=False
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("novelty search failed for %r: %s", card.name, exc)
                continue
            if not hits:
                continue
            top = hits[0]
            payload = top.get("payload") or {}
            score = float(top.get("score") or 0.0)
            if score >= self.novelty_tracing_threshold:
                verdict = "tracing"
            elif score >= self.novelty_borderline_threshold:
                verdict = "borderline"
            else:
                verdict = "novel"
            card.novelty = NoveltyCheck(
                max_similarity=round(score, 4),
                nearest_title=str(payload.get("article_title", "")),
                nearest_url=str(payload.get("article_url", "")),
                verdict=verdict,
            )
            self.on_progress(
                "verify", f"  {card.name}: {verdict} (nearest {score:.3f} — {card.novelty.nearest_title[:40]})"
            )

    def _audit_model_claims(self, cards: list[IdeaCard]) -> None:
        claims: list[tuple[IdeaCard, ComponentEvidence, str]] = []
        for card in cards:
            for component in card.model_components():
                if component.claim:
                    claims.append((card, component, component.claim))
        if not claims:
            self.on_progress("verify", "no model-register claims to audit")
            return

        self.on_progress("verify", f"auditing {len(claims)} model-register claim(s) against the corpus")
        try:
            vectors = self.embedder.embed([claim for _, _, claim in claims])
        except EmbeddingError as exc:
            logger.warning("audit embedding failed: %s", exc)
            self.on_progress("verify", f"audit skipped: {exc}")
            return

        claim_payload: list[dict[str, Any]] = []
        hits_payload: list[dict[str, Any]] = []
        for index, ((card, component, claim), vector) in enumerate(zip(claims, vectors)):
            claim_id = f"c{index}"
            claim_payload.append({"id": claim_id, "concept": card.name, "claim": claim})
            try:
                hits = self.vector_store.search(
                    query_vector=vector, top_k=self.audit_top_k, with_vector=False
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("audit search failed for claim %s: %s", claim_id, exc)
                hits = []
            hits_payload.append(
                {
                    "id": claim_id,
                    "passages": [
                        {
                            "score": round(float(h.get("score") or 0.0), 4),
                            "article_title": (h.get("payload") or {}).get("article_title", ""),
                            "article_url": (h.get("payload") or {}).get("article_url", ""),
                            "text": str((h.get("payload") or {}).get("text", ""))[:600],
                        }
                        for h in hits
                    ],
                }
            )

        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_CRITIC},
                    {
                        "role": "user",
                        "content": prompts.user_audit(claim_payload, hits_payload),
                    },
                ],
                temperature=0.2,
                max_tokens=3072,
            )
        except LLMError as exc:
            logger.warning("audit judgement failed: %s", exc)
            self.on_progress("verify", f"audit judgement failed: {exc}")
            return

        verdicts = {
            str(v.get("id", "")): v
            for v in ((data or {}).get("audits", []) or [])
            if isinstance(v, dict)
        }
        for index, (_card, component, _claim) in enumerate(claims):
            verdict = verdicts.get(f"c{index}")
            if not verdict:
                component.audit = "uncorroborated"
                component.audit_note = "no audit verdict returned"
                continue
            raw_verdict = str(verdict.get("verdict", "")).lower().strip()
            if raw_verdict not in ("corroborated", "contested", "uncorroborated"):
                raw_verdict = "uncorroborated"
            component.audit = raw_verdict
            component.audit_note = str(verdict.get("note", "")).strip()
            self.on_progress("verify", f"  [{raw_verdict}] {component.claim[:70]}")

    def _critique(self, cards: list[IdeaCard]) -> None:
        self.on_progress("critique", f"refereeing {len(cards)} concept(s)")
        payload = [
            {
                "name": c.name,
                "statement": c.statement,
                "composition": c.composition,
                "stakes": c.stakes,
                "objection": c.objection,
                "novelty": {
                    "max_similarity": c.novelty.max_similarity,
                    "verdict": c.novelty.verdict,
                    "nearest_title": c.novelty.nearest_title,
                },
                "components": [
                    {
                        "register": comp.register,
                        "claim": comp.claim,
                        "audit": comp.audit,
                        "audit_note": comp.audit_note,
                        "article_title": comp.article_title,
                    }
                    for comp in c.components
                ],
            }
            for c in cards
        ]
        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_CRITIC},
                    {"role": "user", "content": prompts.user_critique(payload)},
                ],
                temperature=0.3,
                max_tokens=4096,
                retries=2,
            )
        except LLMError as exc:
            logger.warning("critique failed: %s", exc)
            self.on_progress("critique", f"critique failed: {exc}")
            return

        by_name: dict[str, dict] = {}
        for verdict in (data or {}).get("verdicts", []) or []:
            if isinstance(verdict, dict):
                by_name[str(verdict.get("name", "")).strip().lower()] = verdict

        for card in cards:
            verdict = by_name.get(card.name.strip().lower())
            if not verdict:
                card.confidence = 0.5
                card.verdict = "unjudged"
                card.stakes = card.stakes or "(no critique returned)"
                continue

            card.verdict = str(verdict.get("verdict", "")).lower().strip() or "unjudged"
            if card.verdict not in ("keep", "revise", "discard"):
                card.verdict = "unjudged"
            card.tracing_of = str(verdict.get("is_tracing_of", "") or "").strip()

            try:
                card.confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.5))))
            except (TypeError, ValueError):
                card.confidence = 0.5
            referee_score = card.confidence

            caps: list[str] = []
            if card.verdict == "discard":
                if card.confidence > 0.2:
                    card.confidence = 0.2
                caps.append("capped at 0.2: referee ruled discard")
            if card.is_tracing():
                if card.confidence > 0.3:
                    card.confidence = 0.3
                caps.append(f"capped at 0.3: referee named it a tracing of {card.tracing_of!r}")

            if caps:
                card.confidence_note = (
                    f"referee scored {referee_score:.2f}; " + "; ".join(caps)
                )

            reasoning = str(verdict.get("reasoning", "")).strip()
            if reasoning:
                card.objection = f"{card.objection}\n\n**Referee:** {reasoning}".strip()
            self.on_progress(
                "critique",
                f"  {card.name}: {card.verdict} (confidence {card.confidence:.2f}"
                + (f", {card.confidence_note}" if card.confidence_note else "")
                + ")",
            )

    def _summarize(self, run: IdeaRun) -> str:
        self.on_progress("summarize", "writing closing assessment")
        kept = [c for c in run.cards if c.confidence >= 0.5]
        card_payload = [
            {
                "name": c.name,
                "statement": c.statement,
                "confidence": c.confidence,
                "novelty": c.novelty.verdict,
                "contested": c.contested(),
            }
            for c in (kept or run.cards)
        ]
        walk_payload = [
            {**w.to_summary(), "note": w.note} for w in run.walks
        ]
        try:
            data = self.llm.complete_json(
                [
                    {"role": "system", "content": prompts.SYSTEM_SUMMARIZER},
                    {
                        "role": "user",
                        "content": prompts.user_summarize(
                            run.seed, run.brief, walk_payload, card_payload
                        ),
                    },
                ],
                temperature=0.5,
                max_tokens=2048,
            )
        except LLMError as exc:
            logger.warning("summarize failed: %s", exc)
            return f"(closing assessment unavailable: {exc})"
        return str((data or {}).get("summary", "")).strip()


def _best_survivor(run: IdeaRun) -> IdeaCard | None:
    """Highest-confidence card worth re-seeding a walk from.

    Returns None when nothing survived critique, so the caller can close the
    run instead of burning a walk on a concept the referee already rejected.
    Tracings are excluded first — re-seeding from a restatement of existing
    work walks back into territory the corpus already covers.
    """
    survivors = [
        c for c in run.cards
        if c.confidence >= 0.5 and c.verdict != "discard" and not c.is_tracing()
    ]
    for pool in (
        [c for c in survivors if not c.contested()],
        survivors,
        [c for c in run.cards if c.confidence >= 0.5],
    ):
        if pool:
            return max(pool, key=lambda c: c.confidence)
    return None


def _fragment_payload(fragment: Fragment) -> dict[str, Any]:
    return {
        "walk_id": fragment.walk_id,
        "chunk_id": fragment.chunk_id,
        "article_title": fragment.article_title,
        "article_url": fragment.article_url,
        "similarity": round(fragment.similarity, 4),
        "depth": fragment.depth,
        "forced_jump": fragment.forced_jump,
        "text": fragment.text[:700],
    }


def _parse_plans(data: Any, fallback_seed: str) -> list[WalkPlan]:
    raw_walks = (data or {}).get("walks", []) if isinstance(data, dict) else []
    if isinstance(raw_walks, dict):
        raw_walks = [raw_walks]
    plans: list[WalkPlan] = []
    for raw in raw_walks or []:
        if not isinstance(raw, dict):
            continue
        try:
            plans.append(WalkPlan.from_dict(raw))
        except ValueError:
            logger.warning("skipping malformed walk plan: %r", raw)
    if not plans and fallback_seed:
        plans.append(WalkPlan(seed=fallback_seed, rationale="fallback: planner returned no usable walk"))
    return plans


def _validate_components(
    components: list[ComponentEvidence], fragments: dict[str, Fragment]
) -> list[ComponentEvidence]:
    """Drop corpus claims that cite a chunk id the walk never returned.

    Grounding is the whole point of the corpus register, so a fabricated
    citation is worse than no citation: the component is demoted to model
    register and annotated rather than silently kept.
    """
    validated: list[ComponentEvidence] = []
    for component in components:
        if component.register == "corpus":
            source = fragments.get(component.chunk_id)
            if source is None:
                component.register = "model"
                component.audit = "uncorroborated"
                component.audit_note = (
                    f"demoted from corpus register: cited chunk_id "
                    f"{component.chunk_id!r} was not in any walk result"
                )
                component.quote = ""
                validated.append(component)
                continue
            component.article_title = source.article_title
            component.article_url = source.article_url
            component.similarity = source.similarity
            if component.quote and component.quote not in source.text:
                component.quote = _best_quote(component.quote, source.text)
            validated.append(component)
        else:
            component.chunk_id = ""
            component.quote = ""
            validated.append(component)
    return validated


def _best_quote(claimed: str, actual: str) -> str:
    """Return a verbatim excerpt of actual closest to the model's claimed quote."""
    words = claimed.split()[:8]
    if not words:
        return actual[:300]
    needle = " ".join(words)
    position = actual.lower().find(needle.lower())
    if position == -1:
        position = actual.lower().find(words[0].lower())
    if position == -1:
        return actual[:300]
    return actual[position : position + 320]
