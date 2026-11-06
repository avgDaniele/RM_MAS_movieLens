"""Minimal evaluation adapter for the existing movie_rag Resource Assistant.

Query construction does not access private ratings. Evaluation delegates to the
shared movie_rag recommendation evaluator. Keep read-only retrieval results distinct from answer text.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class VisibleMovie:
    movie_id: int
    title: str
    rating: float


@dataclass(frozen=True)
class QueryTemplate:
    template_id: str
    renderer: Callable[[list[VisibleMovie]], str]


def title_rating_v1(movies: list[VisibleMovie]) -> str:
    """Stable, human-readable query. No tags, genres, descriptions or LLM calls."""
    rows = sorted(movies, key=lambda x: x.movie_id)
    if not rows:
        return 'Vorrei dei consigli su film da vedere. Non ho ancora indicato preferenze.'
    lines = ['Consigliami dei film che potrebbero piacermi sulla base delle valutazioni che ho fornito.',
             'Le stelle indicano quanto ho apprezzato ciascun film (scala 0,5-5):']
    for m in rows:
        lines.append(f'- [{m.movie_id}] {m.title}: {m.rating:.1f}/5 stelle')
    lines.extend(['Considera sia le valutazioni alte sia quelle basse.',
                  'Suggerisci film diversi da quelli già indicati.'])
    return '\n'.join(lines)


TEMPLATES: dict[str, QueryTemplate] = {
    'title_rating_v1': QueryTemplate('title_rating_v1', title_rating_v1),
}


def register_template(name: str, renderer: Callable[[list[VisibleMovie]], str]) -> None:
    if not name or name in TEMPLATES:
        raise ValueError('Name empty or already defined')
    TEMPLATES[name] = QueryTemplate(name, renderer)


def make_query(frame: pd.DataFrame, metadata: Mapping[int, str], template_id: str) -> str:
    if template_id not in TEMPLATES:
        raise KeyError(f'Unknown template {template_id}; available {list(TEMPLATES)}')
    movies = [VisibleMovie(int(r.movieId), str(metadata[int(r.movieId)]), float(r.rating))
              for r in frame.itertuples(index=False)]
    return TEMPLATES[template_id].renderer(movies)


def input_digest(frame: pd.DataFrame, metadata: Mapping[int, str], template_id: str) -> str:
    payload = {'template': template_id,
               'items': sorted([{'movieId': int(r.movieId), 'rating': float(r.rating),
                                 'title': str(metadata[int(r.movieId)])}
                                for r in frame.itertuples(index=False)], key=lambda m:m['movieId'])}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def field(item, attr, default=None):
    if isinstance(item, Mapping):
        return item.get(attr, default)
    return getattr(item, attr, default)


def extract_rankings(state, *, catalog: set[int], shown: set[int],
                     target_k: int = 10) -> dict:
    """Use the MCP retrieval result, never parse answer prose for the primary ranking.

    No popularity padding: a short list stays short (observable incomplete result).
    """
    response = field(state, 'search_response')
    matches = field(response, 'results', ()) or ()
    if not isinstance(matches, (list, tuple)):
        raise TypeError('search_response.results is not a list/tuple; adapt contract explicitly')
    raw = []
    unknown = []
    for hit in matches:
        movie_id = field(hit, 'resource_id')
        if movie_id is None:
            raise ValueError('MCP hit without resource_id: cannot evaluate')
        try:
            movie_id = int(movie_id)
        except Exception as exc:
            raise ValueError(f'Non-numeric resource_id: {movie_id!r}') from exc
        score = field(hit, 'score', None)
        rank = field(hit, 'rank', len(raw) + 1)
        if movie_id not in catalog:
            unknown.append(movie_id)
            continue
        raw.append({'movieId': movie_id, 'source_rank': int(rank),
                    'score': float(score) if score is not None else None,
                    'title': str(field(hit, 'title', ''))})
    if unknown:
        raise ValueError(f'MCP returned IDs outside MovieLens catalog: {unknown[:12]}')
    raw.sort(key=lambda r: r['source_rank'])
    candidates=[];seen=set();excluded=[]
    for item in raw:
        mid=item['movieId']
        if mid in shown:
            excluded.append(mid)
            continue
        if mid in seen:
            continue
        seen.add(mid)
        candidates.append(item)
    primary=[]
    for i,r in enumerate(candidates[:target_k],start=1):
        primary.append({'movieId':r['movieId'], 'rank':i, 'score':r['score'], 'title':r['title']})
    ans = field(state, 'final_answer')
    answer_items = field(ans, 'recommendations', ()) or ()
    cited=[];seen_answer=set();unmapped=[]
    for rec in answer_items:
        mid=field(rec, 'resource_id')
        if mid is None:
            mid=field(rec, 'movieId')
        try:mid=int(mid)
        except (ValueError,TypeError):
            unmapped.append(str(mid))
            continue
        if mid in shown or mid not in catalog:
            unmapped.append(str(mid))
            continue
        if mid not in seen_answer:
            cited.append(mid);seen_answer.add(mid)
    # No silent assumption that answer IDs necessarily equal retrieval IDs.
    status = field(state, 'status', 'unknown')
    return {'retrieval':primary, 'answer_ids':cited,
            'answer_unmapped':unmapped, 'raw_retrieved':len(raw),
            'eligible_retrieved':len(candidates), 'shown_excluded':excluded,
            'returned':len(primary), 'shortfall':target_k-len(primary),
            'state_status':str(status),
            'mcp_status':str(field(response,'status','missing')),
            'mcp_query':field(response,'query'),
            'final_answer_text':field(ans,'answer'),
            'response_missing':response is None}


def score_rankings(predictions: pd.DataFrame, *, fixed: pd.DataFrame, extra: pd.DataFrame,
                   private_profile: pd.DataFrame, positive_min=4.0, top_k=10,
                   expected_keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compatibility adapter. New experiments use the shared evaluator directly.

    Legacy callers explicitly declare completed empty runs through expected_keys.
    V2 additionally supplies execution records to distinguish failures from empties.
    """
    from movie_rag.recommendation_benchmark import evaluate_rankings
    records = None
    if expected_keys is not None:
        records = expected_keys[['scenario_id', 'k', 'model']].copy()
        records['status'] = 'completed'
    return evaluate_rankings(predictions, fixed=fixed, extra=extra,
                             private_profile=private_profile, positive_min=positive_min,
                             top_k=top_k, expected_keys=expected_keys, run_records=records)


def macro_by_user(per_scenario: pd.DataFrame, owners: Mapping[str, int]):
    from movie_rag.recommendation_benchmark import macro_by_user as shared_macro
    return shared_macro(per_scenario, owners)
