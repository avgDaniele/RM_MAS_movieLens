"""Explicit experiment entrypoints used by the thin recommendation notebooks.

Importing this module never starts training, embedding, Qdrant or LLM calls.
All persisted runs are bound to the protocol, exact subset, code and inputs.
"""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .recommendation_benchmark import (
    COMPLETE_STATUSES, EMPTY_OBS, KEYS, RANK_COLUMNS, RunCache,
    catalog_coverage, descriptive_resources, digest, evaluate_rankings, file_hash,
    macro_by_user, paired_bootstrap, source_hashes, write_json_once,
)
from .recommendation_models import (
    ExplicitMF, aggregate_scores, combine_semantic_scores, gte_affinities, top_k_ranking, unit,
)


def run_identity(protocol, mode, experiment, extra=None):
    selected = protocol.select(mode)
    return {'protocol_manifest': digest(protocol.manifest), 'config': protocol.config.fingerprint,
            'code': source_hashes(protocol.config.root), 'mode': mode, 'experiment': experiment,
            'selected': selected[['scenario_id', 'userId', 'k']].to_dict('records'),
            'extra': extra or {}}


def run_directory(protocol, identity):
    return (protocol.config.base/'evaluations'/identity['experiment']/identity['mode']/digest(identity))


def _write_csv(path, frame):
    path = Path(path)
    contents = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() != contents:
            raise FileExistsError(f'Different results already exist: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(contents)


def expected_keys(selected, models):
    return pd.DataFrame([{'scenario_id': str(row.scenario_id), 'k': int(row.k), 'model': model}
                         for row in selected.itertuples() for model in sorted(models)], columns=KEYS)


def _add_ranking(rows, sid, k, model, scores, ids, observed, top_k):
    ranked_ids, ranked_scores = top_k_ranking(scores, ids, observed, top_k)
    rows.extend({'scenario_id': str(sid), 'k': int(k), 'model': model, 'movieId': int(mid),
                 'rank': rank, 'score': float(score)}
                for rank, (mid, score) in enumerate(zip(ranked_ids, ranked_scores), 1))


def evaluate_protocol(protocol, predictions, expected, run_records):
    counts = protocol.train.groupby('movieId').size().reindex(protocol.ids, fill_value=0).to_dict()
    c = protocol.config.values
    return evaluate_rankings(predictions, **protocol.private_truth(), expected_keys=expected,
                             run_records=run_records, catalog_ids=protocol.ids, train_counts=counts,
                             positive_min=c['positive_min'], negative_max=c['negative_max'], top_k=c['top_k'])


def save_report(protocol, identity, predictions, expected, records):
    directory = run_directory(protocol, identity)
    # The same expected subset applies to every system, including valid empty lists.
    models = sorted(expected.model.unique())
    declared = expected_keys(protocol.select(identity['mode']), models)
    if set(map(tuple, declared.to_numpy())) != set(map(tuple, expected[KEYS].to_numpy())):
        raise ValueError('Comparison has different scenario/level subsets across systems')
    metrics = evaluate_protocol(protocol, predictions, expected, records)
    owners = protocol.scenarios.set_index('scenario_id').userId.to_dict()
    by_user, macro = macro_by_user(metrics, owners)
    paired = paired_bootstrap(by_user, samples=protocol.config.values['bootstrap_samples'],
                              seed=protocol.config.values['seed'])
    coverage = catalog_coverage(predictions, protocol.ids, expected)
    report = {'identity': identity, 'primary_scheme': 'B', 'primary_level': 10,
              'n_users': int(protocol.select(identity['mode']).userId.nunique()),
              'n_scenarios': int(protocol.select(identity['mode']).scenario_id.nunique()),
              'models': models, 'test_interpretation': protocol.config.values['test_interpretation'],
              'limitations': ['Unjudged items have zero observed gain, not inferred negative preference.',
                             'B targets change across levels; compare systems within each level.',
                             'Judged-only positive rate and popularity buckets are diagnostics.',
                             'Bootstrap resamples users after averaging repetitions.']}
    saved_files = []
    for name, frame in [('rankings', predictions), ('run_records', records), ('metrics_per_scenario', metrics),
                        ('metrics_per_user', by_user), ('macro_A_B', macro), ('paired_deltas', paired),
                        ('catalog_coverage', coverage),
                        ('primary_B_k10', macro[(macro.scheme == 'B') & (macro.k == 10)])]:
        _write_csv(directory/f'{name}.csv', frame)
        saved_files.append(f'{name}.csv')
    report['files'] = {name: file_hash(directory/name) for name in saved_files}
    write_json_once(directory/'report_manifest.json', report)
    return {'directory': directory, 'metrics': metrics, 'per_user': by_user, 'macro': macro,
            'paired': paired, 'coverage': coverage}


def _embedding_identity(protocol, resources):
    return {'protocol': digest(protocol.manifest), 'embedding': protocol.config.values['embedding'],
            'documents': digest([{'id': int(r.resource_id), 'text': r.text} for r in resources]),
            'fields': 'TMDb descriptive allowlist; no MovieLens tags or statistics'}


def _resources(protocol):
    frame = pd.read_csv(protocol.config.root/protocol.config.values['catalog_path'])
    frame = frame[frame.movieId.isin(protocol.ids)].sort_values('movieId')
    resources = descriptive_resources(frame)
    if [int(r.resource_id) for r in resources] != protocol.ids.tolist():
        raise ValueError('Descriptive resources do not cover the common catalog')
    return resources


def _validate_vectors(ids, vectors, expected_ids, dimension):
    if not np.array_equal(ids, expected_ids) or vectors.shape != (len(expected_ids), dimension):
        raise ValueError('Embedding IDs/dimensions do not match the common catalog')
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-4):
        raise ValueError('Invalid or unnormalized embeddings')


def load_vectors(protocol):
    directory = protocol.config.base/'artifacts/embeddings'
    manifest = json.loads((directory/'manifest.json').read_text())
    resources = _resources(protocol)
    if manifest['identity'] != _embedding_identity(protocol, resources):
        raise ValueError('Embedding representation or input fingerprint changed')
    if file_hash(directory/'vectors.npz') != manifest['vectors_hash']:
        raise ValueError('Embedding cache changed')
    with np.load(directory/'vectors.npz', allow_pickle=False) as saved:
        ids, vectors = saved['movie_ids'], saved['vectors']
    _validate_vectors(ids, vectors, protocol.ids, protocol.config.values['embedding']['dimension'])
    return vectors, digest(manifest)


def verify_collection(client, protocol, embedding_fingerprint):
    """Reject a partial/stale collection before any pipeline predictions are made."""
    embedding = protocol.config.values['embedding']
    seen, offset = set(), None
    while True:
        points, offset = client.scroll(collection_name=embedding['collection_name'], limit=256,
                                       offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            payload = point.payload or {}
            metadata = payload.get('metadata') or {}
            mid = int(payload.get('resource_id', -1))
            if (mid in seen or metadata.get('benchmark_embedding_fingerprint') != embedding_fingerprint or
                    payload.get('embedding_model') != embedding['model_name']):
                raise ValueError('Incompatible or duplicate benchmark collection point')
            seen.add(mid)
        if offset is None:
            break
    if seen != set(protocol.ids):
        raise ValueError('Benchmark collection does not exactly cover the common catalog')


def prepare_embeddings(protocol, *, encoder=None, client=None):
    """Explicit, potentially expensive operation. Never deletes an existing collection."""
    from .resource_embeddings import DenseEncoderSpec, SentenceTransformerResourceEncoder
    from .resource_qdrant import QdrantCollectionConfig, connect_qdrant, ensure_collection, upsert_resources
    import os

    resources = _resources(protocol)
    identity = _embedding_identity(protocol, resources)
    embedding = protocol.config.values['embedding']
    directory = protocol.config.base/'artifacts/embeddings'
    saved = directory/'vectors.npz'
    if saved.exists():
        vectors, fingerprint = load_vectors(protocol)
    else:
        encoder = encoder or SentenceTransformerResourceEncoder(DenseEncoderSpec(
            name='coldstart_no_interactions', model_name=embedding['model_name'],
            dimension=embedding['dimension'], trust_remote_code=True))
        vectors = np.asarray(encoder.encode_documents([r.text for r in resources]), dtype=np.float32)
        _validate_vectors(protocol.ids, vectors, protocol.ids, embedding['dimension'])
        directory.mkdir(parents=True, exist_ok=True)
        with saved.open('xb') as stream:
            np.savez_compressed(stream, movie_ids=protocol.ids, vectors=vectors)
        manifest = {'identity': identity, 'vectors_hash': file_hash(saved)}
        write_json_once(directory/'manifest.json', manifest)
        fingerprint = digest(manifest)
    client = client or connect_qdrant(os.getenv('QDRANT_URL', 'http://localhost:6333'), os.getenv('QDRANT_API_KEY'))
    collection = QdrantCollectionConfig(embedding['collection_name'], embedding['dimension'],
                                         embedding['vector_name'], schema_version='coldstart_no_interactions_v1')
    existing = {c.name for c in client.get_collections().collections}
    if collection.collection_name in existing:
        # Permit resuming a partial upsert only if every existing point has this provenance.
        offset = None
        while True:
            points, offset = client.scroll(collection_name=collection.collection_name, limit=256,
                                           offset=offset, with_payload=True, with_vectors=False)
            for point in points:
                if ((point.payload or {}).get('metadata') or {}).get('benchmark_embedding_fingerprint') != fingerprint:
                    raise ValueError('Refusing to overwrite an incompatible existing collection')
            if offset is None:
                break
    ensure_collection(client, collection, recreate=False)
    resources = [r.model_copy(update={'metadata': {'benchmark_embedding_fingerprint': fingerprint}})
                 for r in resources]
    upsert_resources(client, collection, resources, vectors, embedding['model_name'])
    verify_collection(client, protocol, fingerprint)
    return fingerprint


def baseline_candidates(config):
    candidates = {
        'bayesian_mean': {'family': 'bayesian_mean', 'params': {}},
        'most_popular': {'family': 'most_popular', 'params': {}},
    }
    mf = config.values['mf']
    # A true bias-only model (rank zero), tuned under the same selection rule.
    for family, ranks in [('mf_explicit_foldin', mf['ranks']), ('mf_bias_only', [0])]:
        for rank, epochs, reg, fold in product(ranks, mf['epochs'], mf['regularization'], mf['fold_regularization']):
            params = {'rank': rank, 'epochs': epochs, 'reg': reg, 'fold_reg': fold}
            key = family + ':' + json.dumps(params, sort_keys=True, separators=(',', ':'))
            candidates[key] = {'family': family, 'params': params}
    candidates.update(semantic_candidates(config, 'gte'))
    return candidates


def semantic_candidates(config, prefix):
    grid = config.values['semantic']
    candidates = {}
    for family, weights in [(f'{prefix}_direct', [1.]), (f'{prefix}_prior', grid['semantic_weights'])]:
        for alpha, negative_only, weight in product(grid['mixed_negative_alpha'], grid['negative_only_alpha'], weights):
            params = {'mixed_alpha': alpha, 'negative_only_alpha': negative_only, 'semantic_weight': weight}
            key = family + ':' + json.dumps(params, sort_keys=True, separators=(',', ':'))
            candidates[key] = {'family': family, 'params': params}
    return candidates


def generate_baselines(protocol, selected, candidates, vectors):
    """Training uses only protocol.train. Private targets are not accessed here."""
    c = protocol.config.values
    bayesian, counts = aggregate_scores(protocol.train, protocol.ids, c['bayesian_prior'])
    observed = protocol.observed
    rows = []
    # Group equivalent fits; fold-in choices reuse a single trained model.
    fits = {}
    for spec in candidates.values():
        family, params = spec['family'], spec['params']
        if family.startswith('mf_'):
            fit_key = (params['rank'], params['epochs'], params['reg'])
            if fit_key not in fits:
                fits[fit_key] = ExplicitMF(protocol.ids, rank=params['rank'], epochs=params['epochs'],
                    reg=params['reg'], lr=c['mf']['learning_rate'], seed=c['seed']).fit(protocol.train)
    for rec in selected.itertuples():
        obs = observed.get((str(rec.scenario_id), int(rec.k)), EMPTY_OBS)
        # Cosines do not depend on alpha or the prior weight. Compute them only once per profile.
        positive, negative = gte_affinities(obs, protocol.ids, vectors, c['positive_min'], c['negative_max'])
        for model, spec in candidates.items():
            family, params = spec['family'], spec['params']
            if family == 'most_popular':
                scores = counts
            elif family == 'bayesian_mean' or rec.k == 0:
                scores = bayesian
            elif family.startswith('mf_'):
                fit = fits[(params['rank'], params['epochs'], params['reg'])]
                scores = fit.score(obs, params['fold_reg'])
            else:
                scores = combine_semantic_scores(positive, negative, bayesian, **params)
            _add_ranking(rows, rec.scenario_id, rec.k, model, scores, protocol.ids, obs, c['top_k'])
    return pd.DataFrame(rows, columns=RANK_COLUMNS)


def choose_on_validation(protocol, selected, predictions, candidates):
    if set(selected.split) != {'validation'}:
        raise ValueError('Hyperparameters can only be selected on validation')
    expected = expected_keys(selected, candidates)
    records = expected.assign(status='completed')
    metrics = evaluate_protocol(protocol, predictions, expected, records)
    by_user, macro = macro_by_user(metrics, protocol.scenarios.set_index('scenario_id').userId.to_dict())
    del by_user
    tune = macro[(macro.scheme == 'B') & (macro.k > 0)].groupby('model')['ndcg@10'].mean()
    winners = {}
    for family in sorted({spec['family'] for spec in candidates.values()}):
        keys = [key for key in candidates if candidates[key]['family'] == family]
        scores = tune.reindex(keys).dropna()
        if scores.empty:
            raise ValueError(f'No defined validation scores for {family}')
        winner = sorted(scores.index, key=lambda key: (-scores[key], key))[0]
        winners[family] = {'candidate_id': winner, **candidates[winner], 'validation_ndcg_B': float(scores[winner])}
    return winners, macro


def _selection_path(protocol, experiment):
    return protocol.config.base/'selections'/f'{experiment}.json'


def _selection_identity(protocol, extra):
    return {'protocol': digest(protocol.manifest), 'code': source_hashes(protocol.config.root), 'extra': extra}


def load_selection(protocol, experiment, extra):
    selection = json.loads(_selection_path(protocol, experiment).read_text())
    if selection['identity'] != _selection_identity(protocol, extra):
        raise ValueError('Frozen validation selection is incompatible with this run')
    return selection


def run_baselines(protocol, mode='pilot_validation'):
    vectors, embedding_hash = load_vectors(protocol)  # GTE must be present; no partial comparisons
    selected = protocol.select(mode)
    extra = {'embedding': embedding_hash}
    if mode == 'test':
        selection = load_selection(protocol, 'baselines', extra)
        candidates = {family: {'family': family, 'params': spec['params']}
                      for family, spec in selection['winners'].items()}
    else:
        candidates = baseline_candidates(protocol.config)
    predictions = generate_baselines(protocol, selected, candidates, vectors)
    identity = run_identity(protocol, mode, 'baselines', extra)
    if mode != 'test':
        winners, macro = choose_on_validation(protocol, selected, predictions, candidates)
        _write_csv(run_directory(protocol, identity)/'validation_candidates.csv', macro)
        if mode == 'full_validation':
            write_json_once(_selection_path(protocol, 'baselines'),
                            {'identity': _selection_identity(protocol, extra), 'winners': winners})
        chosen = {spec['candidate_id']: family for family, spec in winners.items()}
        predictions = predictions[predictions.model.isin(chosen)].copy()
        predictions['model'] = predictions.model.map(chosen)
    models = set(predictions.model)
    expected = expected_keys(selected, models)
    return save_report(protocol, identity, predictions, expected, expected.assign(status='completed'))


class SemanticProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    positive_query: str | None = Field(default=None, max_length=600)
    negative_query: str | None = Field(default=None, max_length=600)
    brief_rationale: str = Field(max_length=350)


SEMANTIC_PROMPT = """Represent ONLY the revealed movie preferences as compact ENGLISH retrieval queries.
positive_query: themes, style and genres of films rated >=4.
negative_query: descriptive themes and style of films rated <=2, without negation or 'avoid'.
Ratings 2.5–3.5 are neutral. Return null for an absent class. Do not invent hidden ratings,
recommend titles, infer demographic attributes, or generalize unsupported preferences.
Return the structured profile with a brief rationale."""


def _llm_context(protocol, provider=None):
    from .llm_config import load_llm_config
    from .llm_providers import LLMProviderRegistry
    cfg = protocol.config.values['llm']
    path = protocol.config.root/cfg['config_path']
    catalog = load_llm_config(path)
    profile = catalog.get_llm_profile(cfg['profile_id'])
    if provider is None:
        provider = LLMProviderRegistry(catalog).get(cfg['profile_id'])
    return provider, {'config_hash': file_hash(path), 'profile': profile.model_dump(mode='json'),
                      'temperature': cfg['temperature']}


def visible_payload(protocol, obs):
    movies = pd.read_csv(protocol.config.root/protocol.config.values['movies_path']).set_index('movieId')
    return [{'movie_id': int(r.movieId), 'title': str(movies.loc[r.movieId, 'title']),
             'genres': str(movies.loc[r.movieId, 'genres']), 'rating': float(r.rating)}
            for r in obs.sort_values('movieId').itertuples()]


def _cache_key(sid, k, payload):
    return digest({'scenario_id': str(sid), 'k': int(k), 'visible': payload})


def validate_semantic_profile(profile, obs, positive_min=4., negative_max=2.):
    # Dictionaries from caches are validated just as strictly as new provider output.
    profile = SemanticProfile.model_validate(profile)
    for field in ['positive_query', 'negative_query']:
        text = getattr(profile, field)
        setattr(profile, field, text.strip() or None if isinstance(text, str) else None)
    if (bool(profile.positive_query) != bool((obs.rating >= positive_min).any()) or
            bool(profile.negative_query) != bool((obs.rating <= negative_max).any())):
        raise ValueError('Semantic queries do not match revealed rating classes')
    return profile


async def generate_semantic_profiles(protocol, mode, provider, identity, max_new_calls):
    selected, observed = protocol.select(mode), protocol.observed
    write_json_once(run_directory(protocol, identity)/'run_identity.json', identity)
    cache = RunCache(run_directory(protocol, identity)/'semantic_profiles.jsonl', identity)
    pending = []
    for row in selected.itertuples():
        obs = observed.get((str(row.scenario_id), int(row.k)), EMPTY_OBS)
        payload = visible_payload(protocol, obs)
        key = _cache_key(row.scenario_id, row.k, payload)
        if key not in cache.rows and ((obs.rating >= 4) | (obs.rating <= 2)).any():
            pending.append(key)
    if len(pending) > max_new_calls:
        raise ValueError(f'{len(pending)} new semantic calls exceed budget {max_new_calls}')
    profiles = {}
    for row in selected.itertuples():
        sid, k = str(row.scenario_id), int(row.k)
        obs = observed.get((sid, k), EMPTY_OBS)
        payload = visible_payload(protocol, obs)
        key = _cache_key(sid, k, payload)
        if key in cache.rows:
            value = cache.rows[key]['payload']
        else:
            trace = None
            try:
                if ((obs.rating >= 4) | (obs.rating <= 2)).any():
                    result = await provider.generate_structured(
                        agent='coldstart_semantic_profile', prompt_version='coldstart_semantic_v2',
                        system_prompt=SEMANTIC_PROMPT, user_prompt=json.dumps(payload, ensure_ascii=False),
                        output_schema=SemanticProfile, temperature=protocol.config.values['llm']['temperature'],
                        max_output_tokens=700)
                    profile = validate_semantic_profile(result.value, obs)
                    trace = result.trace.model_dump(mode='json') if getattr(result, 'trace', None) else None
                else:
                    profile = SemanticProfile(brief_rationale='No non-neutral revealed feedback')
                value = {'profile': profile.model_dump(), 'trace': trace}
                cache.append(key, 'completed', value)
            except Exception as exc:
                cache.append(key, 'error', {'exception_type': type(exc).__name__, 'scenario_id': sid, 'k': k})
                raise RuntimeError(f'Incomplete semantic run: resume {sid}/{k}') from exc
        profiles[(sid, k)] = validate_semantic_profile(value['profile'], obs).model_dump()
    return profiles


def generate_semantic_rankings(protocol, selected, profiles, query_vectors, vectors, candidates):
    bayesian, _ = aggregate_scores(protocol.train, protocol.ids, protocol.config.values['bayesian_prior'])
    observed, rows = protocol.observed, []
    affinities = {query: vectors @ unit(vector) for query, vector in query_vectors.items()}
    for row in selected.itertuples():
        sid, k = str(row.scenario_id), int(row.k)
        obs = observed.get((sid, k), EMPTY_OBS)
        profile = validate_semantic_profile(profiles[(sid, k)], obs)
        pos = affinities.get(profile.positive_query)
        neg = affinities.get(profile.negative_query)
        if (profile.positive_query and pos is None) or (profile.negative_query and neg is None):
            raise ValueError('Missing query embedding')
        for model, spec in candidates.items():
            scores = combine_semantic_scores(pos, neg, bayesian, **spec['params'])
            _add_ranking(rows, sid, k, model, scores, protocol.ids, obs, 10)
    return pd.DataFrame(rows, columns=RANK_COLUMNS)


def load_baseline_comparison(protocol, mode, selected):
    _, embedding_hash = load_vectors(protocol)
    identity = run_identity(protocol, mode, 'baselines', {'embedding': embedding_hash})
    directory = run_directory(protocol, identity)
    manifest = json.loads((directory/'report_manifest.json').read_text())
    if manifest['identity'] != identity:
        raise ValueError('Incompatible baseline report')
    for name, expected_hash in manifest['files'].items():
        if file_hash(directory/name) != expected_hash:
            raise ValueError(f'Baseline report changed: {name}')
    predictions = pd.read_csv(directory/'rankings.csv')
    records = pd.read_csv(directory/'run_records.csv')
    wanted = set(map(tuple, selected[['scenario_id', 'k']].to_numpy()))
    if set(map(tuple, predictions[['scenario_id', 'k']].drop_duplicates().to_numpy())) != wanted:
        raise ValueError('Baseline comparison subset mismatch')
    return predictions, records


async def run_semantic_llm(protocol, mode='pilot_validation', *, provider=None, encoder=None, max_new_calls=None):
    from .resource_embeddings import DenseEncoderSpec, SentenceTransformerResourceEncoder
    vectors, embedding_hash = load_vectors(protocol)
    provider, llm = _llm_context(protocol, provider)
    extra = {'embedding': embedding_hash, 'llm': llm, 'prompt': digest(SEMANTIC_PROMPT)}
    identity = run_identity(protocol, mode, 'llm_semantic', extra)
    selected = protocol.select(mode)
    # Require a compatible baseline before spending any API budget.
    baseline, baseline_records = load_baseline_comparison(protocol, mode, selected)
    if mode == 'test':
        selection = load_selection(protocol, 'llm_semantic', extra)
        candidates = {family: {'family': family, 'params': spec['params']}
                      for family, spec in selection['winners'].items()}
    else:
        candidates = semantic_candidates(protocol.config, 'llm_gte')
    profiles = await generate_semantic_profiles(protocol, mode, provider, identity,
        protocol.config.values['llm']['max_new_calls'] if max_new_calls is None else max_new_calls)
    queries = sorted({value for profile in profiles.values()
                      for field, value in profile.items() if field.endswith('_query') and value})
    encoder = encoder or SentenceTransformerResourceEncoder(DenseEncoderSpec(
        name='coldstart_query', model_name=protocol.config.values['embedding']['model_name'],
        dimension=vectors.shape[1], trust_remote_code=True))
    encoded = encoder.encode_documents(queries) if queries else np.empty((0, vectors.shape[1]))
    if encoded.shape != (len(queries), vectors.shape[1]) or not np.isfinite(encoded).all():
        raise ValueError('Query/corpus embedding mismatch')
    predictions = generate_semantic_rankings(protocol, selected, profiles, dict(zip(queries, encoded)), vectors, candidates)
    if mode != 'test':
        winners, macro = choose_on_validation(protocol, selected, predictions, candidates)
        _write_csv(run_directory(protocol, identity)/'validation_candidates.csv', macro)
        if mode == 'full_validation':
            write_json_once(_selection_path(protocol, 'llm_semantic'),
                            {'identity': _selection_identity(protocol, extra), 'winners': winners})
        chosen = {spec['candidate_id']: family for family, spec in winners.items()}
        predictions = predictions[predictions.model.isin(chosen)].copy()
        predictions['model'] = predictions.model.map(chosen)
    expected = expected_keys(selected, predictions.model.unique())
    records = expected.assign(status='completed')
    return save_report(protocol, identity, pd.concat([baseline, predictions], ignore_index=True),
                       pd.concat([baseline_records[KEYS], expected], ignore_index=True),
                       pd.concat([baseline_records, records], ignore_index=True))


def _pipeline_query(payload):
    lines = ['Consigliami film in base ai soli rating rivelati (scala 0.5–5).',
             'Considera sia voti alti sia bassi; suggerisci film diversi da quelli elencati.']
    lines += [f"- [{row['movie_id']}] {row['title']}: {row['rating']:.1f}/5" for row in payload]
    return '\n'.join(lines)


def extract_pipeline_ranking(state, catalog, shown, top_k=10):
    """Accept valid empty results; reject incomplete/error states and unknown IDs."""
    status = state.status
    if status not in COMPLETE_STATUSES:
        raise ValueError(f'Incomplete pipeline state: {status}')
    response = state.search_response
    if response is None and status != 'needs_clarification':
        raise ValueError('Completed search is missing its response')
    hits = [] if response is None else response.results
    if status in {'needs_clarification', 'no_results'} and hits:
        raise ValueError('Empty state contains search hits')
    primary, seen = [], set()
    for hit in sorted(hits, key=lambda h: h.rank):
        mid = int(hit.resource_id)
        if mid not in catalog:
            raise ValueError('Pipeline returned item outside common catalog')
        if mid in shown or mid in seen:
            continue
        if not np.isfinite(hit.score):
            raise ValueError('Invalid retrieval score')
        seen.add(mid)
        primary.append({'movieId': mid, 'rank': len(primary)+1, 'score': float(hit.score)})
    return primary[:top_k]


async def execute_pipeline(protocol, mode, workflow, identity, max_new_calls):
    selected, observed = protocol.select(mode), protocol.observed
    write_json_once(run_directory(protocol, identity)/'run_identity.json', identity)
    cache = RunCache(run_directory(protocol, identity)/'workflow_runs.jsonl', identity)
    jobs = []
    for row in selected.itertuples():
        obs = observed.get((str(row.scenario_id), int(row.k)), EMPTY_OBS)
        payload = visible_payload(protocol, obs)
        jobs.append((row, obs, payload, _cache_key(row.scenario_id, row.k, payload)))
    if sum(key not in cache.rows and row.k > 0 for row, _, _, key in jobs) > max_new_calls:
        raise ValueError('New pipeline calls exceed the declared budget')
    bayesian, _ = aggregate_scores(protocol.train, protocol.ids, protocol.config.values['bayesian_prior'])
    rows, records = [], []
    model = 'llm_pipeline_linear_retrieval'
    for row, obs, payload, key in jobs:
        sid, k = str(row.scenario_id), int(row.k)
        if key in cache.rows:
            result = cache.rows[key]
        else:
            try:
                start = perf_counter()
                if k == 0:
                    ids, scores = top_k_ranking(bayesian, protocol.ids, obs)
                    primary = [{'movieId': int(mid), 'rank': rank, 'score': float(score)}
                               for rank, (mid, score) in enumerate(zip(ids, scores), 1)]
                    status, state = 'k0_bayesian', None
                else:
                    state = await workflow.run(_pipeline_query(payload))
                    primary = extract_pipeline_ranking(state, set(protocol.ids), set(obs.movieId))
                    status = state.status
                detail = {'ranking': primary, 'latency_seconds': perf_counter()-start,
                          'state': state.model_dump(mode='json') if state is not None else None}
                result = cache.append(key, status, detail)
            except Exception as exc:
                cache.append(key, 'error', {'exception_type': type(exc).__name__, 'scenario_id': sid, 'k': k})
                raise RuntimeError(f'Incomplete pipeline run: resume {sid}/{k}') from exc
        records.append({'scenario_id': sid, 'k': k, 'model': model, 'status': result['status']})
        rows.extend({'scenario_id': sid, 'k': k, 'model': model, **hit} for hit in result['payload']['ranking'])
    return pd.DataFrame(rows, columns=RANK_COLUMNS), pd.DataFrame(records)


async def run_pipeline(protocol, mode='pilot_validation', *, max_new_calls=None):
    from mcp import Client
    from .resource_domain import load_resource_domain_catalog
    from .resource_mcp_profiles import ResourceSearchServiceRegistry
    from .resource_mcp import create_resource_mcp_server
    from .resource_mcp_gateway import MCPResourceToolGateway
    from .resource_llm_workflow import build_resource_assistant_workflow
    from .resource_adapters import movie_filters_to_resource_filters

    _, embedding_hash = load_vectors(protocol)
    selected = protocol.select(mode)
    baseline, baseline_records = load_baseline_comparison(protocol, mode, selected)
    provider, llm = _llm_context(protocol)
    cfg = protocol.config.values
    domain_path = protocol.config.root/'configs/resource_domains_v1.json'
    mcp_path = protocol.config.root/'configs/resource_mcp_profiles_v1.json'
    domain = load_resource_domain_catalog(domain_path).get(cfg['pipeline']['domain_id'])
    registry = ResourceSearchServiceRegistry.from_environment(mcp_path, profile_id=cfg['pipeline']['profile_id'])
    profile = registry.profile
    search = registry.service.profiles[profile.search_profile_id]
    if (domain.allowed_filter_keys or domain.domain_id != profile.domain_id or not profile.exact or
            profile.collection_name != cfg['embedding']['collection_name'] or profile.reranker is not None or
            search.candidate_k != 30 or search.final_k != 30 or search.rerank or
            profile.encoder.model_name != cfg['embedding']['model_name']):
        raise ValueError('Benchmark requires the exact unfiltered 30-candidate profile and matching encoder')
    verify_collection(registry.client, protocol, embedding_hash)
    extra = {'embedding': embedding_hash, 'llm': llm, 'domain_hash': file_hash(domain_path),
             'mcp_hash': file_hash(mcp_path), 'template': digest(_pipeline_query([{'movie_id': 0, 'title': '', 'rating': 0.}]))}
    if mode == 'test':
        load_selection(protocol, 'pipeline', extra)
    identity = run_identity(protocol, mode, 'pipeline', extra)
    server = create_resource_mcp_server(registry.service, filter_adapter=movie_filters_to_resource_filters,
        default_profile_id=profile.search_profile_id, expose_profile_to_client=False, exact=True)
    async with Client(server, raise_exceptions=True) as client:
        workflow = build_resource_assistant_workflow(domain=domain, planner_provider=provider,
            answer_provider=provider, gateway=MCPResourceToolGateway(client), result_limit=30,
            planner_temperature=cfg['llm']['temperature'], answer_temperature=cfg['llm']['temperature'],
            query_rewrite_strategy='translate_to_catalog_language')
        predictions, records = await execute_pipeline(protocol, mode, workflow, identity,
            cfg['llm']['max_new_calls'] if max_new_calls is None else max_new_calls)
    if mode == 'full_validation':
        write_json_once(_selection_path(protocol, 'pipeline'), {'identity': _selection_identity(protocol, extra)})
    expected = expected_keys(selected, ['llm_pipeline_linear_retrieval'])
    return save_report(protocol, identity, pd.concat([baseline, predictions], ignore_index=True),
                       pd.concat([baseline_records[KEYS], expected], ignore_index=True),
                       pd.concat([baseline_records, records], ignore_index=True))


def read_reports(protocol, mode):
    """Read only compatible reports; do not concatenate repeated baseline comparisons."""
    reports = []
    for path in sorted((protocol.config.base/'evaluations').glob(f'*/{mode}/*/report_manifest.json')):
        manifest = json.loads(path.read_text())
        identity = manifest['identity']
        if identity['protocol_manifest'] != digest(protocol.manifest) or identity['code'] != source_hashes(protocol.config.root):
            raise ValueError(f'Incompatible report: {path}')
        for name, expected_hash in manifest['files'].items():
            if file_hash(path.parent/name) != expected_hash:
                raise ValueError(f'Report changed: {path.parent/name}')
        frame = pd.read_csv(path.parent/'macro_A_B.csv')
        frame['experiment'] = identity['experiment']
        frame['mode'] = mode
        reports.append(frame)
    return pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()


def read_traces(protocol, mode):
    """Include incomplete runs while checking their protocol, subset and code identity."""
    rows = []
    wanted = protocol.select(mode)[['scenario_id', 'userId', 'k']].to_dict('records')
    for path in sorted((protocol.config.base/'evaluations').glob(f'*/{mode}/*/run_identity.json')):
        identity = json.loads(path.read_text())
        if (identity['protocol_manifest'] != digest(protocol.manifest) or
                identity['code'] != source_hashes(protocol.config.root) or identity['selected'] != wanted):
            raise ValueError(f'Incompatible trace identity: {path}')
        for name in ['semantic_profiles.jsonl', 'workflow_runs.jsonl']:
            cache_path = path.parent/name
            if not cache_path.exists():
                continue
            # Validate even error records; the diagnostic view keeps the history of retries.
            RunCache(cache_path, identity)
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                rows.append({'experiment': identity['experiment'], 'mode': mode, **row})
    return pd.DataFrame(rows)
