"""Versioned cold-start protocol, provenance and the single recommendation evaluator.

This module is deliberately independent of model providers and Qdrant. Missing
ratings are unjudged, never inferred dislikes. B is primary; A is legacy only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

KEYS = ['scenario_id', 'k', 'model']
RANK_COLUMNS = KEYS + ['movieId', 'rank', 'score']
EMPTY_OBS = pd.DataFrame(columns=['movieId', 'rating'])
COMPLETE_STATUSES = {'completed', 'no_results', 'needs_clarification', 'k0_bayesian'}
METRICS = ['ndcg@10', 'recall@10', 'hr@10', 'n_hits_topk', 'n_returned', 'shortfall',
           'n_judged_topk', 'n_unjudged_topk', 'n_negative_topk', 'n_neutral_topk',
           'judged_coverage@10', 'positive_rate_judged']
BUCKETS = ['zero', 'tail_1_4', 'midtail_5_19', 'mid_20_99', 'head_100plus']
PUBLIC_FIELDS = (
    ('Title', 'tmdb_title'), ('genres', 'tmdb_genres'), ('Overview', 'overview'),
    ('Tagline', 'tagline'), ('keywords', 'tmdb_keyword_text'), ('Directors', 'directors'),
    ('Writers', 'writers'), ('Composers', 'composers'), ('Main cast', 'cast_top'),
    ('Main characters', 'cast_characters_top'), ('Collection', 'collection_name'),
    ('Production companies', 'production_companies'),
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes(root):
    paths = sorted((Path(root)/'src/movie_rag').rglob('*.py'))
    paths += sorted((Path(root)/'src/movie_rag/prompts').glob('*.txt'))
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def write_json_once(path, value):
    path = Path(path)
    text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Incompatible immutable artifact: {path}; use a new version')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)


@dataclass(frozen=True)
class BenchmarkConfig:
    root: Path
    values: dict

    @classmethod
    def load(cls, root, path='configs/recommendation_coldstart_v2.json'):
        root = Path(root).resolve()
        values = json.loads((root/path).read_text())
        required = {'protocol_id', 'source_protocol', 'seed', 'levels', 'profile_size',
                    'fixed_holdout_size', 'minimum_history', 'repetitions', 'positive_min',
                    'negative_max', 'top_k', 'embedding', 'mf', 'semantic', 'pipeline'}
        if not required.issubset(values):
            raise ValueError(f'Missing benchmark configuration: {required-set(values)}')
        if (values['profile_size'] != 10 or values['fixed_holdout_size'] != 20 or
                values['levels'] != [0, 1, 3, 5, 10] or values['top_k'] != 10 or
                values['minimum_history'] != 30 or values['repetitions'] != 3 or
                values['train_fraction'] != .8 or values['validation_fraction'] != .1 or
                values['positive_min'] != 4 or values['negative_max'] != 2 or
                values['primary_scheme'] != 'B' or values['primary_level'] != 10):
            raise ValueError('Configuration does not implement the frozen cold-start design')
        if values['protocol_id'] == values['source_protocol']:
            raise ValueError('A new protocol ID is required')
        if values['pipeline']['retrieval_limit'] < values['top_k'] + values['profile_size']:
            raise ValueError('Insufficient retrieval candidates for visible-item removal')
        return cls(root, values)

    @property
    def base(self):
        return self.root/'datasets/processed/recommendation_protocol'/self.values['protocol_id']

    @property
    def fingerprint(self):
        return digest(self.values)


def _ids(frame, cols):
    for col in cols:
        numeric = pd.to_numeric(frame[col], errors='raise')
        if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f'Invalid integer identifier: {col}')
        frame[col] = numeric.astype('int64')


def validate_protocol(items, train, scenarios, levels, visible, profile, fixed, extra, partitions, cfg):
    """Validate provenance-independent invariants before exporting any v2 inputs."""
    for frame in [items, train, visible, profile, fixed, extra]:
        _ids(frame, ['movieId'])
    for frame in [train, scenarios, levels, profile, fixed, extra, partitions]:
        _ids(frame, ['userId'])
    for frame in [visible, levels]:
        _ids(frame, ['k'])
    _ids(profile, ['reveal_order'])
    catalog = set(items.movieId)
    if items.movieId.duplicated().any() or not catalog:
        raise ValueError('Catalog must be nonempty and unique')
    for frame in [train, visible, profile, fixed, extra]:
        if not set(frame.movieId).issubset(catalog):
            raise ValueError('Item outside common catalog')
        ratings = pd.to_numeric(frame.rating, errors='raise').to_numpy(float)
        if not np.isfinite(ratings).all() or not np.isin(ratings, np.arange(.5, 5.1, .5)).all():
            raise ValueError('Invalid MovieLens rating')
    if partitions.userId.duplicated().any() or not set(partitions.split).issubset({'train', 'validation', 'test'}):
        raise ValueError('Invalid user partitions')
    strata = partitions.groupby('eligible') if 'eligible' in partitions else [(None, partitions)]
    for _, stratum in strata:
        n = len(stratum)
        n_val = int(round(n * cfg['validation_fraction']))
        n_test = int(round(n * (1-cfg['train_fraction']-cfg['validation_fraction'])))
        observed_counts = stratum.split.value_counts().to_dict()
        if any(observed_counts.get(name, 0) != count for name, count in
               [('validation', n_val), ('test', n_test), ('train', n-n_val-n_test)]):
            raise ValueError('Source partitions do not implement the 80/10/10 split')
    owners = partitions.set_index('userId').split.to_dict()
    train_users = {uid for uid, split in owners.items() if split == 'train'}
    if set(train.userId) != train_users or train[['userId', 'movieId']].duplicated().any():
        raise ValueError('Training does not match source partitions')
    if scenarios.scenario_id.duplicated().any() or set(train.userId) & set(scenarios.userId):
        raise ValueError('Duplicate scenarios or train/evaluation user leakage')
    if scenarios.empty or not set(scenarios.split).issubset({'validation', 'test'}):
        raise ValueError('Invalid evaluation scenarios')
    for row in scenarios.itertuples():
        if owners.get(row.userId) != row.split:
            raise ValueError('Scenario owner/split differs from partitions')
    reps = scenarios.groupby('userId').rep.apply(lambda x: sorted(x.tolist()))
    if not reps.apply(lambda x: x == list(range(cfg['repetitions']))).all():
        raise ValueError('Missing or duplicate repetitions')
    expected_levels = {(sid, k) for sid in scenarios.scenario_id for k in cfg['levels']}
    if (levels[['scenario_id', 'k']].duplicated().any() or
            set(map(tuple, levels[['scenario_id', 'k']].to_numpy())) != expected_levels):
        raise ValueError('Missing/duplicate scenario levels')
    scenario_owners = scenarios.set_index('scenario_id').userId.to_dict()
    scenario_splits = scenarios.set_index('scenario_id').split.to_dict()
    scenario_reps = scenarios.set_index('scenario_id').rep.to_dict()
    for row in levels.itertuples():
        if (row.userId != scenario_owners[row.scenario_id] or row.split != scenario_splits[row.scenario_id]
                or row.rep != scenario_reps[row.scenario_id]):
            raise ValueError('Inconsistent level metadata')
    if visible[['scenario_id', 'k', 'movieId']].duplicated().any():
        raise ValueError('Duplicate visible interactions')
    if not set(map(tuple, visible[['scenario_id', 'k']].to_numpy())).issubset(expected_levels):
        raise ValueError('Unknown visible scenario/level')
    for frame in [profile, fixed, extra]:
        if frame[['scenario_id', 'movieId']].duplicated().any() or not set(frame.scenario_id).issubset(scenario_owners):
            raise ValueError('Duplicate/unknown private targets')
        if any(row.userId != scenario_owners[row.scenario_id] for row in frame.itertuples()):
            raise ValueError('Private target owner mismatch')
    profile_groups = dict(tuple(profile.groupby('scenario_id')))
    fixed_groups = dict(tuple(fixed.groupby('scenario_id')))
    extra_groups = dict(tuple(extra.groupby('scenario_id')))
    visible_groups = dict(tuple(visible.groupby(['scenario_id', 'k'])))
    history_by_user = {}
    for row in scenarios.itertuples():
        p = profile_groups.get(row.scenario_id, profile.iloc[:0]).sort_values('reveal_order')
        f = fixed_groups.get(row.scenario_id, fixed.iloc[:0])
        e = extra_groups.get(row.scenario_id, extra.iloc[:0])
        if len(p) != 10 or len(f) != 20 or p.reveal_order.tolist() != list(range(1, 11)):
            raise ValueError('Expected profile of 10 and fixed holdout of 20')
        sets = [set(frame.movieId) for frame in [p, f, e]]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError('Overlapping profile/fixed/additional sets')
        history = dict(zip(pd.concat([p, f, e]).movieId, pd.concat([p, f, e]).rating))
        if row.userId in history_by_user and history != history_by_user[row.userId]:
            raise ValueError('User history differs across repetitions')
        history_by_user[row.userId] = history
        for k in cfg['levels']:
            observed = visible_groups.get((row.scenario_id, k), visible.iloc[:0])
            if (len(observed) != k or dict(zip(observed.movieId, observed.rating)) !=
                    dict(zip(p.head(k).movieId, p.head(k).rating))):
                raise ValueError('Visible interactions are not exact profile prefixes')
    if 'n_ratings' in partitions:
        actual_history = train.groupby('userId').size().to_dict()
        actual_history.update({uid: len(history) for uid, history in history_by_user.items()})
        for row in partitions.itertuples():
            if row.userId in actual_history and row.n_ratings != actual_history[row.userId]:
                raise ValueError('Source history size differs from the copied protocol')


def prepare_protocol(config):
    """Copy and validate the existing full protocol, preserving its exact random splits."""
    c = config.values
    source = config.root/'datasets/processed/recommendation_protocol'/c['source_protocol']
    old_manifest = json.loads((source/'protocol_manifest.json').read_text())
    checks = {'protocol_id': c['source_protocol'], 'seed': c['seed'],
              'profile_size': c['profile_size'], 'fixed_holdout_size': c['fixed_holdout_size'],
              'levels': c['levels'], 'repetitions': c['repetitions'],
              'positive_rating_min': c['positive_min'], 'negative_rating_max': c['negative_max']}
    if any(old_manifest.get(key) != value for key, value in checks.items()):
        raise ValueError('Source protocol differs from the frozen design')
    relative = [f'inputs/{name}.csv' for name in ['catalog_movie_ids', 'train_ratings', 'scenarios',
                                                'scenario_levels', 'visible_interactions']]
    relative += [f'private_evaluation/{name}.csv' for name in ['user_partitions', 'private_profile',
                                                            'private_fixed_gt', 'private_additional']]
    frames = {name: pd.read_csv(source/name) for name in relative}
    get = lambda name: frames[next(path for path in frames if Path(path).stem == name)]
    validate_protocol(get('catalog_movie_ids'), get('train_ratings'), get('scenarios'),
                      get('scenario_levels'), get('visible_interactions'), get('private_profile'),
                      get('private_fixed_gt'), get('private_additional'), get('user_partitions'), c)
    catalog = pd.read_csv(config.root/c['catalog_path'])
    allowed = set(catalog.loc[catalog.tmdb_enrichment_status != 'missing_tmdb', 'movieId'].astype(int))
    if allowed != set(get('catalog_movie_ids').movieId):
        raise ValueError('Source catalog IDs differ from the descriptive catalog')
    source_catalog_hash = old_manifest.get('sources', {}).get('catalog_sha256')
    if source_catalog_hash != file_hash(config.root/c['catalog_path']):
        raise ValueError('Catalog differs from the source protocol snapshot')
    manifest = {'protocol_id': c['protocol_id'], 'config': c, 'config_hash': config.fingerprint,
                'source_manifest_hash': file_hash(source/'protocol_manifest.json'),
                'files': {name: file_hash(source/name) for name in relative},
                'catalog_hash': file_hash(config.root/c['catalog_path']),
                'movies_hash': file_hash(config.root/c['movies_path']),
                'code': source_hashes(config.root), 'primary_scheme': 'B', 'primary_level': 10,
                'test_interpretation': c['test_interpretation']}
    # Preflight every destination before writing any file; never replace old artifacts.
    for name in relative:
        dest = config.base/name
        if dest.exists() and file_hash(dest) != manifest['files'][name]:
            raise FileExistsError(f'Incompatible protocol file: {dest}')
    manifest_path = config.base/'protocol_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise FileExistsError('Protocol manifest changed; use a new protocol ID')
    for name in relative:
        dest = config.base/name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as stream:
                stream.write((source/name).read_bytes())
    write_json_once(manifest_path, manifest)
    return Protocol.load(config)


@dataclass
class Protocol:
    config: BenchmarkConfig
    manifest: dict
    items: pd.DataFrame
    train: pd.DataFrame
    scenarios: pd.DataFrame
    levels: pd.DataFrame
    visible: pd.DataFrame

    @classmethod
    def load(cls, config):
        manifest = json.loads((config.base/'protocol_manifest.json').read_text())
        if manifest['config_hash'] != config.fingerprint or manifest['code'] != source_hashes(config.root):
            raise ValueError('Protocol/config/code fingerprint mismatch; prepare a new version')
        for name, expected in manifest['files'].items():
            if file_hash(config.base/name) != expected:
                raise ValueError(f'Protocol input changed: {name}')
        for key, path in [('catalog_hash', 'catalog_path'), ('movies_hash', 'movies_path')]:
            if file_hash(config.root/config.values[path]) != manifest[key]:
                raise ValueError(f'External input changed: {path}')
        names = ['catalog_movie_ids', 'train_ratings', 'scenarios', 'scenario_levels', 'visible_interactions']
        return cls(config, manifest, *(pd.read_csv(config.base/f'inputs/{name}.csv') for name in names))

    @property
    def ids(self):
        return np.sort(self.items.movieId.to_numpy(dtype=np.int64))

    @property
    def observed(self):
        return {(str(s), int(k)): g[['movieId', 'rating']].copy()
                for (s, k), g in self.visible.groupby(['scenario_id', 'k'])}

    def select(self, mode):
        if mode not in {'pilot_validation', 'full_validation', 'test'}:
            raise ValueError('Unknown run mode')
        subset = self.scenarios[self.scenarios.split == ('test' if mode == 'test' else 'validation')]
        if mode == 'pilot_validation':
            subset = subset[subset.rep == 0]
            users = np.sort(subset.userId.unique())
            chosen = np.random.default_rng(self.config.values['seed']).choice(
                users, min(len(users), self.config.values['pilot_users']), replace=False)
            subset = subset[subset.userId.isin(chosen)]
        selected = self.levels[self.levels.scenario_id.isin(subset.scenario_id)].copy()
        if selected.empty:
            raise ValueError('No scenarios selected')
        return selected.sort_values(['scenario_id', 'k']).reset_index(drop=True)

    def private_truth(self):
        # Called only once the predictions have been computed/frozen.
        return {key: pd.read_csv(self.config.base/f'private_evaluation/{name}.csv')
                for key, name in [('fixed', 'private_fixed_gt'), ('extra', 'private_additional'),
                                  ('private_profile', 'private_profile')]}


def popularity_bucket(count):
    return ('zero' if count == 0 else 'tail_1_4' if count <= 4 else
            'midtail_5_19' if count <= 19 else 'mid_20_99' if count <= 99 else 'head_100plus')


def evaluate_rankings(predictions, *, fixed, extra, private_profile, positive_min=4.,
                      negative_max=2., top_k=10, expected_keys=None, run_records=None,
                      catalog_ids=None, train_counts=None):
    """Single A/B evaluator. Explicit completed records are required for empty lists."""
    if top_k != 10:
        raise ValueError('The benchmark reports metrics at cutoff 10')
    required = set(KEYS + ['movieId', 'rank'])
    if not required.issubset(predictions):
        raise ValueError(f'Missing ranking columns: {required-set(predictions)}')
    if predictions[list(required)].isna().any().any():
        raise ValueError('Null ranking identifiers')
    predictions = predictions.copy()
    _ids(predictions, ['movieId', 'rank', 'k'])
    truth = {}
    for name, frame in [('F', fixed), ('E', extra), ('P', private_profile)]:
        for sid, group in frame.groupby('scenario_id'):
            if group.movieId.duplicated().any():
                raise ValueError('Duplicate truth item')
            truth.setdefault(str(sid), {})[name] = [(int(r.movieId), float(r.rating),
                int(r.reveal_order) if name == 'P' else 0) for r in group.itertuples()]
    actual = set(map(tuple, predictions[KEYS].itertuples(index=False, name=None)))
    expected = actual if expected_keys is None else set(map(tuple, expected_keys[KEYS].itertuples(index=False, name=None)))
    if expected_keys is not None and expected_keys[KEYS].duplicated().any():
        raise ValueError('Duplicate expected keys')
    if not actual.issubset(expected):
        raise ValueError('Unexpected ranking scenario/level/model')
    if run_records is not None:
        if run_records[KEYS].duplicated().any():
            raise ValueError('Duplicate run records')
        records = {tuple(row[key] for key in KEYS): row['status'] for row in run_records.to_dict('records')}
        if set(records) != expected or any(status not in COMPLETE_STATUSES for status in records.values()):
            raise ValueError('Incomplete run: resume infrastructure failures before evaluation')
        for key in actual:
            if records[key] in {'no_results', 'needs_clarification'}:
                raise ValueError('Empty-result state contains ranking rows')
    elif expected - actual:
        raise ValueError('Missing rankings require explicit completed run records')
    catalog = None if catalog_ids is None else set(map(int, catalog_ids))
    counts = {} if train_counts is None else train_counts
    groups = {key: group for key, group in predictions.groupby(KEYS)}
    rows = []
    for sid, k, model in sorted(expected):
        if int(k) not in (0, 1, 3, 5, 10) or str(sid) not in truth:
            raise ValueError('Unknown scenario/level')
        records = truth[str(sid)]
        f, e, p = (records.get(name, []) for name in ('F', 'E', 'P'))
        if len({mid for mid, _, _ in f+e+p}) != len(f+e+p):
            raise ValueError('Overlapping private truth sets')
        shown = {mid for mid, _, order in p if order <= k}
        group = groups.get((sid, k, model), predictions.iloc[:0]).sort_values('rank')
        if (len(group) > top_k or group.movieId.duplicated().any() or
                group['rank'].tolist() != list(range(1, len(group)+1))):
            raise ValueError('Malformed ranking')
        ids = group.movieId.astype(int).tolist()
        if set(ids) & shown or (catalog is not None and not set(ids).issubset(catalog)):
            raise ValueError('Revealed or out-of-catalog recommendation')
        for scheme, entries in [('A', f), ('B', f+e+[row for row in p if row[2] > k])]:
            ratings = {mid: rating for mid, rating, _ in entries}
            pos = {mid for mid, rating in ratings.items() if rating >= positive_min}
            labels = np.asarray([mid in pos for mid in ids], dtype=float)
            npos, hits = len(pos), int(labels.sum())
            weights = 1/np.log2(np.arange(2, len(ids)+2))
            ideal = (1/np.log2(np.arange(2, min(top_k, npos)+2))).sum()
            judged = sum(mid in ratings for mid in ids)
            result = {'scenario_id': str(sid), 'k': int(k), 'model': str(model), 'scheme': scheme,
                      'n_relevant': npos, 'n_hits_topk': hits,
                      'ndcg@10': float(labels @ weights / ideal) if npos else np.nan,
                      'recall@10': hits/npos if npos else np.nan,
                      'hr@10': float(hits > 0) if npos else np.nan,
                      'n_returned': len(ids), 'shortfall': top_k-len(ids),
                      'n_judged_topk': judged, 'n_unjudged_topk': len(ids)-judged,
                      'n_negative_topk': sum(mid in ratings and ratings[mid] <= negative_max for mid in ids),
                      'n_neutral_topk': sum(mid in ratings and negative_max < ratings[mid] < positive_min for mid in ids),
                      'judged_coverage@10': judged/top_k,
                      'positive_rate_judged': hits/judged if judged else np.nan}
            if train_counts is not None:
                if catalog is None or not set(counts).issuperset(catalog):
                    raise ValueError('Training frequency must cover the full catalog, including zero-count items')
                for bucket in BUCKETS:
                    targets = {mid for mid in pos if popularity_bucket(counts[mid]) == bucket}
                    result[f'n_relevant_{bucket}'] = len(targets)
                    result[f'recall_{bucket}@10'] = len(set(ids) & targets)/len(targets) if targets else np.nan
            rows.append(result)
    return pd.DataFrame(rows)


def macro_by_user(per_scenario, owners: Mapping[str, int]):
    frame = per_scenario.copy()
    frame['userId'] = frame.scenario_id.map(owners)
    if frame.userId.isna().any():
        raise ValueError('Missing user mapping')
    metrics = [name for name in frame if name in METRICS or name.startswith('recall_') or name.startswith('n_relevant')]
    by_user = frame.groupby(['userId', 'k', 'model', 'scheme'], as_index=False)[metrics].mean()
    macro = by_user.groupby(['k', 'model', 'scheme'], as_index=False)[metrics].mean()
    n = by_user.groupby(['k', 'model', 'scheme'])['ndcg@10'].count().rename('n_users_defined').reset_index()
    return by_user, macro.merge(n, on=['k', 'model', 'scheme'], validate='one_to_one')


def paired_bootstrap(per_user, baseline='bayesian_mean', samples=2000, seed=20261021):
    rows = []
    rng = np.random.default_rng(seed)
    for (scheme, k), frame in per_user.groupby(['scheme', 'k']):
        for metric in ['ndcg@10', 'recall@10', 'hr@10']:
            pivot = frame.pivot(index='userId', columns='model', values=metric)
            if baseline not in pivot:
                raise ValueError('Missing paired Bayesian baseline')
            for model in sorted(set(pivot)-{baseline}):
                pair = pivot[[baseline, model]].dropna()
                differences = (pair[model]-pair[baseline]).to_numpy(float)
                if len(differences) >= 2:
                    bootstrap = np.array([rng.choice(differences, len(differences), replace=True).mean()
                                          for _ in range(samples)])
                    low, high = np.quantile(bootstrap, [.025, .975])
                else:
                    low = high = np.nan
                rows.append({'scheme': scheme, 'k': int(k), 'metric': metric, 'model': model,
                             'baseline': baseline, 'n_users': len(pair),
                             'delta': float(differences.mean()) if len(pair) else np.nan,
                             'ci95_low': low, 'ci95_high': high})
    return pd.DataFrame(rows)


def catalog_coverage(predictions, catalog_ids, selected):
    denominator = len(set(catalog_ids))
    rows = []
    for (model, k), group in predictions.groupby(['model', 'k']):
        rows.append({'model': model, 'k': int(k), 'unique_recommended': group.movieId.nunique(),
                     'catalog_coverage': group.movieId.nunique()/denominator})
    # Empty systems still appear with zero coverage.
    for row in selected[KEYS].drop_duplicates().itertuples(index=False):
        if not any(r['model'] == row.model and r['k'] == row.k for r in rows):
            rows.append({'model': row.model, 'k': int(row.k), 'unique_recommended': 0, 'catalog_coverage': 0.})
    return pd.DataFrame(rows)


def descriptive_resources(frame):
    from .documents import clean_text_value
    from .resource_models import ResourceDocument
    if frame.movieId.duplicated().any():
        raise ValueError('Duplicate descriptive catalog ID')
    resources = []
    for row in frame.to_dict('records'):
        title = clean_text_value(row.get('tmdb_title')) or clean_text_value(row.get('canonical_title'))
        if not title:
            raise ValueError('Missing public movie title')
        fields = {column: row.get(column) for _, column in PUBLIC_FIELDS}
        fields['tmdb_title'] = title
        text = '\n'.join(f'{label}: {clean_text_value(fields[column])}' for label, column in PUBLIC_FIELDS
                         if clean_text_value(fields[column]))
        resources.append(ResourceDocument(resource_id=int(row['movieId']), title=title, text=text,
                                          domain='movielens_coldstart_v2', metadata={}))
    return resources


class RunCache:
    """Fingerprint-bound JSONL cache. Failures are logged but never reused as results."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.fingerprint = digest(identity)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if row.get('fingerprint') != self.fingerprint:
                    raise ValueError(f'Incompatible run cache: {self.path}')
                if row['status'] in COMPLETE_STATUSES:
                    self.rows[row['key']] = row

    def append(self, key, status, payload):
        row = {'key': key, 'status': status, 'fingerprint': self.fingerprint, 'payload': payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        if status in COMPLETE_STATUSES:
            self.rows[key] = row
        return row
