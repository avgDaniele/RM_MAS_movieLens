"""Synthetic regression tests only: no real dataset, Qdrant, encoder or LLM calls."""
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from movie_rag.recommendation_benchmark import (
    BenchmarkConfig, BUCKETS, KEYS, RANK_COLUMNS, Protocol, RunCache,
    descriptive_resources, digest, evaluate_rankings, macro_by_user, paired_bootstrap,
    prepare_protocol, validate_protocol,
)
from movie_rag.recommendation_experiments import (
    SemanticProfile, baseline_candidates, choose_on_validation, execute_pipeline,
    expected_keys, extract_pipeline_ranking, generate_baselines, generate_semantic_rankings,
    load_selection, run_identity, save_report, validate_semantic_profile,
    load_vectors, prepare_embeddings, read_reports, read_traces, run_baselines, run_semantic_llm,
)
from movie_rag.recommendation_models import ExplicitMF, aggregate_scores, gte_scores, top_k_ranking

REPO = Path(__file__).resolve().parents[1]


def fixture(root):
    c = json.loads((REPO/'configs/recommendation_coldstart_v2.json').read_text())
    c['bootstrap_samples'] = 30
    config = BenchmarkConfig(Path(root), c)
    ids = np.arange(1, 51)
    train = pd.DataFrame([(u, m, float(1+m % 5)) for u in range(1, 9) for m in range(1, 31)],
                         columns=['userId', 'movieId', 'rating'])
    scenarios, profile, fixed, extra, levels, visible = [], [], [], [], [], []
    for uid, split in [(9, 'validation'), (10, 'test')]:
        for rep in range(3):
            sid = f'{split}_u{uid}_r{rep}'
            scenarios.append((sid, uid, split, rep))
            history = np.roll(np.arange(1, 36), rep*3)
            for order, mid in enumerate(history, 1):
                row = (sid, uid, int(mid), float(1+mid % 5))
                if order <= 10:
                    profile.append((*row, order))
                elif order <= 30:
                    fixed.append(row)
                else:
                    extra.append(row)
            for k in c['levels']:
                levels.append((sid, uid, split, rep, k))
                for order, mid in enumerate(history[:k], 1):
                    visible.append((sid, k, int(mid), float(1+mid % 5), order))
    frames = dict(
        items=pd.DataFrame({'movieId': ids}), train=train,
        scenarios=pd.DataFrame(scenarios, columns=['scenario_id', 'userId', 'split', 'rep']),
        levels=pd.DataFrame(levels, columns=['scenario_id', 'userId', 'split', 'rep', 'k']),
        visible=pd.DataFrame(visible, columns=['scenario_id', 'k', 'movieId', 'rating', 'reveal_order']),
        profile=pd.DataFrame(profile, columns=['scenario_id', 'userId', 'movieId', 'rating', 'reveal_order']),
        fixed=pd.DataFrame(fixed, columns=['scenario_id', 'userId', 'movieId', 'rating']),
        extra=pd.DataFrame(extra, columns=['scenario_id', 'userId', 'movieId', 'rating']),
        partitions=pd.DataFrame({'userId': range(1, 11), 'split': ['train']*8+['validation', 'test']}),
    )
    public = pd.DataFrame({'movieId': ids, 'tmdb_title': [f'Public {m}' for m in ids],
                          'canonical_title': [f'Title {m}' for m in ids],
                          'tmdb_genres': 'Drama', 'overview': 'Public plot',
                          'tmdb_enrichment_status': 'complete', 'ml_tags': 'SECRET_TAG',
                          'ml_rating_mean': 5., 'ml_tag_text': 'SECRET_TAG_TEXT',
                          'text_combined': 'CONTAMINATED_TEXT'})
    path = Path(root)/c['catalog_path']; path.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(path, index=False)
    path = Path(root)/c['movies_path']; path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'movieId': ids, 'title': [f'Title {m}' for m in ids], 'genres': 'Drama'}).to_csv(path, index=False)
    protocol = Protocol(config, {'synthetic': True}, *(frames[name] for name in
        ['items', 'train', 'scenarios', 'levels', 'visible']))
    protocol.private_truth = lambda: {key: frames[name] for key, name in
                                      [('fixed', 'fixed'), ('extra', 'extra'), ('private_profile', 'profile')]}
    return protocol, frames


class TestProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p, self.f = fixture(self.temp.name)

    def validate(self, frames=None):
        f = self.f if frames is None else frames
        validate_protocol(*(f[name] for name in ['items', 'train', 'scenarios', 'levels', 'visible',
                                                'profile', 'fixed', 'extra', 'partitions']), self.p.config.values)

    def test_full_synthetic_protocol_and_pilot(self):
        self.validate()
        self.assertEqual(len(self.p.select('pilot_validation')), 5)
        self.assertEqual(len(self.p.select('full_validation')), 15)
        self.assertEqual(set(self.p.select('test').split), {'test'})

    def test_leakage_wrong_prefix_and_overlap_rejected(self):
        for name, column, value in [('train', 'userId', 9), ('visible', 'rating', .5),
                                     ('fixed', 'movieId', int(self.f['profile'].movieId.iloc[0]))]:
            with self.subTest(name=name):
                frames = {key: value.copy() for key, value in self.f.items()}
                frames[name].loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.validate(frames)

    def test_public_documents_ignore_every_interaction_field(self):
        public = pd.read_csv(self.p.config.root/self.p.config.values['catalog_path'])
        first = descriptive_resources(public)
        public['ml_tags'] = 'OTHER_SECRET'
        public['ml_rating_mean'] = 1.
        public['text_combined'] = 'OTHER_CONTAMINATION'
        second = descriptive_resources(public)
        self.assertEqual([r.text for r in first], [r.text for r in second])
        self.assertFalse(any('SECRET' in r.text or 'CONTAMINAT' in r.text for r in first))

    def test_prepare_preserves_source_files_and_rejects_mutation(self):
        from movie_rag.recommendation_benchmark import file_hash
        c = self.p.config.values
        source = self.p.config.root/'datasets/processed/recommendation_protocol'/c['source_protocol']
        mapping = {'inputs/catalog_movie_ids.csv': 'items', 'inputs/train_ratings.csv': 'train',
                   'inputs/scenarios.csv': 'scenarios', 'inputs/scenario_levels.csv': 'levels',
                   'inputs/visible_interactions.csv': 'visible', 'private_evaluation/user_partitions.csv': 'partitions',
                   'private_evaluation/private_profile.csv': 'profile', 'private_evaluation/private_fixed_gt.csv': 'fixed',
                   'private_evaluation/private_additional.csv': 'extra'}
        for rel, name in mapping.items():
            path = source/rel; path.parent.mkdir(parents=True, exist_ok=True)
            self.f[name].to_csv(path, index=False)
        old = {'protocol_id': c['source_protocol'], 'seed': c['seed'], 'profile_size': 10,
               'fixed_holdout_size': 20, 'levels': c['levels'], 'repetitions': 3,
               'positive_rating_min': 4., 'negative_rating_max': 2.,
               'sources': {'catalog_sha256': file_hash(self.p.config.root/c['catalog_path'])}}
        (source/'protocol_manifest.json').write_text(json.dumps(old))
        prepared = prepare_protocol(self.p.config)
        self.assertTrue(np.array_equal(prepared.ids, self.p.ids))
        for rel in mapping:
            self.assertEqual((source/rel).read_bytes(), (self.p.config.base/rel).read_bytes())
        prepare_protocol(self.p.config)  # idempotent, no overwrite
        with (self.p.config.base/'inputs/train_ratings.csv').open('a') as stream:
            stream.write('999,1,5\n')
        with self.assertRaises(ValueError):
            Protocol.load(self.p.config)
        with self.assertRaises(FileExistsError):
            prepare_protocol(self.p.config)

    def test_cache_rejects_input_prompt_model_code_and_representation_changes(self):
        path = Path(self.temp.name)/'run.jsonl'
        identity = dict(input='one', prompt='one', model='one', code='one', representation='one')
        cache = RunCache(path, identity)
        cache.append('key', 'completed', {'answer': []})
        for key in identity:
            changed = dict(identity); changed[key] = 'changed'
            with self.subTest(key=key), self.assertRaises(ValueError):
                RunCache(path, changed)
        cache.append('failed', 'error', {'exception_type': 'TimeoutError'})
        self.assertNotIn('failed', RunCache(path, identity).rows)


class TestModels(unittest.TestCase):
    def test_untrained_item_never_has_random_latent_effect(self):
        train = pd.DataFrame({'userId': [1, 1, 2, 2], 'movieId': [1, 2, 1, 2], 'rating': [5., 1., 4., 2.]})
        mf = ExplicitMF([1, 2, 3], rank=8, epochs=2).fit(train)
        np.testing.assert_array_equal(mf.Q[2], np.zeros(8))
        self.assertEqual(mf.item_bias[2], 0)
        ub, factors = mf.fold_in(pd.DataFrame({'movieId': [3], 'rating': [5.]}))
        self.assertGreater(ub, 0)
        np.testing.assert_array_equal(factors, np.zeros(8))
        self.assertEqual(mf.score(pd.DataFrame({'movieId': [1], 'rating': [5.]}))[2],
                         mf.mu + mf.fold_in(pd.DataFrame({'movieId': [1], 'rating': [5.]}))[0])

    def test_bias_only_and_aggregate_baselines_are_distinct(self):
        train = pd.DataFrame({'userId': [1, 2, 3, 1], 'movieId': [1, 1, 1, 2], 'rating': [1., 1., 1., 5.]})
        bayes, counts = aggregate_scores(train, [1, 2, 3])
        self.assertGreater(bayes[1], bayes[0])
        self.assertGreater(counts[0], counts[1])
        mf = ExplicitMF([1, 2, 3], rank=0, epochs=2).fit(train)
        self.assertEqual(mf.Q.shape, (3, 0))
        self.assertTrue(np.isfinite(mf.score(pd.DataFrame({'movieId': [3], 'rating': [1.]}))).all())

    def test_seen_items_removed_before_cutoff_and_ties_by_id(self):
        ids = np.arange(1, 41)
        observed = pd.DataFrame({'movieId': range(1, 11)})
        ranked, _ = top_k_ranking(np.ones(40), ids, observed)
        self.assertEqual(ranked.tolist(), list(range(11, 21)))

    def test_semantic_prior_and_negative_only_alpha_are_effective(self):
        vectors = np.eye(3)
        ids = [1, 2, 3]; prior = np.array([0., 1., 3.])
        positive = pd.DataFrame({'movieId': [1], 'rating': [5.]})
        pure = gte_scores(positive, ids, vectors, prior, semantic_weight=1.)
        mixed = gte_scores(positive, ids, vectors, prior, semantic_weight=0.)
        self.assertEqual(np.argmax(pure), 0)
        self.assertEqual(np.argmax(mixed), 2)
        negative = pd.DataFrame({'movieId': [3], 'rating': [1.]})
        a = gte_scores(negative, ids, vectors, prior, negative_only_alpha=0.)
        b = gte_scores(negative, ids, vectors, prior, negative_only_alpha=.7)
        self.assertLess(b[2], a[2])


class TestEvaluation(unittest.TestCase):
    def truth(self):
        fixed = pd.DataFrame({'scenario_id': ['s', 's'], 'movieId': [1, 2], 'rating': [5., 1.]})
        extra = pd.DataFrame({'scenario_id': ['s'], 'movieId': [3], 'rating': [5.]})
        profile = pd.DataFrame({'scenario_id': ['s', 's'], 'movieId': [4, 5], 'rating': [5., 3.], 'reveal_order': [1, 2]})
        return dict(fixed=fixed, extra=extra, private_profile=profile)

    def test_hand_computed_A_B_unknown_negative_neutral(self):
        ids = [3, 1, 2, 5, 6]
        pred = pd.DataFrame([dict(scenario_id='s', k=1, model='m', movieId=mid, rank=i)
                             for i, mid in enumerate(ids, 1)])
        metrics = evaluate_rankings(pred, **self.truth(), catalog_ids=range(1, 7),
                                    train_counts={i: 0 for i in range(1, 7)}).set_index('scheme')
        self.assertAlmostEqual(metrics.loc['A', 'ndcg@10'], 1/np.log2(3))
        self.assertAlmostEqual(metrics.loc['B', 'ndcg@10'], 1.)
        self.assertEqual(metrics.loc['B', 'recall@10'], 1.)
        self.assertEqual(metrics.loc['B', 'n_unjudged_topk'], 1)
        self.assertEqual(metrics.loc['B', 'n_negative_topk'], 1)
        self.assertEqual(metrics.loc['B', 'n_neutral_topk'], 1)
        self.assertEqual(metrics.loc['B', 'shortfall'], 5)
        self.assertEqual(metrics.loc['B', 'recall_zero@10'], 1.)

    def test_empty_is_zero_only_when_run_completed(self):
        pred = pd.DataFrame(columns=RANK_COLUMNS)
        keys = pd.DataFrame([dict(scenario_id='s', k=1, model='m')])
        with self.assertRaises(ValueError):
            evaluate_rankings(pred, **self.truth(), expected_keys=keys)
        for status in ['no_results', 'needs_clarification', 'completed']:
            out = evaluate_rankings(pred, **self.truth(), expected_keys=keys,
                                    run_records=keys.assign(status=status))
            self.assertTrue(out['ndcg@10'].eq(0).all())
            self.assertTrue(out.shortfall.eq(10).all())
        for status in ['error', 'created', 'missing']:
            with self.subTest(status=status), self.assertRaises(ValueError):
                evaluate_rankings(pred, **self.truth(), expected_keys=keys, run_records=keys.assign(status=status))

    def test_no_positives_is_undefined_and_seen_rejected(self):
        truth = self.truth()
        for frame in truth.values():
            frame['rating'] = 1.
        pred = pd.DataFrame([dict(scenario_id='s', k=1, model='m', movieId=1, rank=1)])
        out = evaluate_rankings(pred, **truth)
        self.assertTrue(out['ndcg@10'].isna().all())
        pred['movieId'] = 4
        with self.assertRaises(ValueError):
            evaluate_rankings(pred, **truth)

    def test_legacy_notebook_adapter_uses_identical_metrics(self):
        import sys
        sys.path.insert(0, str(REPO/'notebooks'))
        from recommendation_05_pipeline_helpers import score_rankings
        pred = pd.DataFrame([dict(scenario_id='s', k=1, model='m', movieId=3, rank=1)])
        pd.testing.assert_frame_equal(evaluate_rankings(pred, **self.truth()), score_rankings(pred, **self.truth()))

    def test_repetitions_are_averaged_before_user_bootstrap(self):
        rows = []
        for sid, user, value in [('a', 1, 0.), ('b', 1, 0.), ('c', 2, 1.)]:
            for model, score in [('bayesian_mean', 0.), ('m', value)]:
                rows.append(dict(scenario_id=sid, model=model, k=10, scheme='B', **{
                    'ndcg@10': score, 'recall@10': score, 'hr@10': score}))
        by_user, macro = macro_by_user(pd.DataFrame(rows), {'a': 1, 'b': 1, 'c': 2})
        self.assertEqual(macro.set_index('model').loc['m', 'ndcg@10'], .5)
        paired = paired_bootstrap(by_user, samples=50)
        self.assertTrue(paired.n_users.eq(2).all())
        self.assertTrue(paired.delta.eq(.5).all())


class TestExperiments(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.p, self.f = fixture(self.temp.name)

    def test_validation_selection_is_B_only_and_ties_are_lexical(self):
        selected = self.p.select('pilot_validation')
        candidates = {name: {'family': 'f', 'params': {}} for name in ['a_legacy', 'c_full', 'b_full']}
        rows = []
        for row in selected.itertuples():
            # A prefers a_legacy (one fixed positive). B prefers b_full/c_full
            # (two additional positives). The lexical tie must choose b_full.
            for model, ids in [('a_legacy', [14]), ('b_full', [33, 34]), ('c_full', [33, 34])]:
                for rank, mid in enumerate(ids, 1):
                    rows.append(dict(scenario_id=row.scenario_id, k=row.k, model=model,
                                     movieId=mid, rank=rank, score=1.))
        winners, _ = choose_on_validation(self.p, selected, pd.DataFrame(rows), candidates)
        self.assertEqual(winners['f']['candidate_id'], 'b_full')
        with self.assertRaises(ValueError):
            choose_on_validation(self.p, self.p.select('test'), pd.DataFrame(rows), candidates)
        with self.assertRaises(FileNotFoundError):
            load_selection(self.p, 'baselines', {})

    def test_synthetic_baselines_share_k0_and_report_all_diagnostics(self):
        selected = self.p.select('pilot_validation')
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(50, 3)); vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        candidates = {
            'bayesian_mean': {'family': 'bayesian_mean', 'params': {}},
            'most_popular': {'family': 'most_popular', 'params': {}},
            'mf': {'family': 'mf_explicit_foldin', 'params': {'rank': 8, 'epochs': 2, 'reg': .1, 'fold_reg': 1.}},
            'gte': {'family': 'gte_direct', 'params': {}},
        }
        pred = generate_baselines(self.p, selected, candidates, vectors)
        zero = pred[(pred.k == 0) & (pred.model != 'most_popular')].groupby('model').movieId.apply(list)
        self.assertTrue(all(ids == zero.iloc[0] for ids in zero))
        keys = expected_keys(selected, candidates)
        result = save_report(self.p, run_identity(self.p, 'pilot_validation', 'test_synthetic'),
                             pred, keys, keys.assign(status='completed'))
        self.assertTrue((result['directory']/'primary_B_k10.csv').exists())
        self.assertIn('judged_coverage@10', result['macro'])
        self.assertTrue(all(f'recall_{bucket}@10' in result['macro'] for bucket in BUCKETS))
        # Reproducible report writes are idempotent.
        save_report(self.p, run_identity(self.p, 'pilot_validation', 'test_synthetic'),
                    pred, keys, keys.assign(status='completed'))

    def test_pipeline_thirty_hits_keep_ten_unseen(self):
        state = NS(status='completed', search_response=NS(results=[
            NS(resource_id=i, rank=i, score=1/i) for i in range(1, 31)]))
        ranking = extract_pipeline_ranking(state, set(range(1, 41)), set(range(1, 11)))
        self.assertEqual([r['movieId'] for r in ranking], list(range(11, 21)))
        self.assertEqual([r['rank'] for r in ranking], list(range(1, 11)))
        state.status = 'error'
        with self.assertRaises(ValueError):
            extract_pipeline_ranking(state, set(range(1, 41)), set())

    def test_pipeline_valid_empty_is_retained_and_error_is_resumable(self):
        class EmptyWorkflow:
            async def run(self, query):
                return NS(status='no_results', search_response=NS(results=[]), model_dump=lambda **_: {'status': 'no_results'})
        identity = run_identity(self.p, 'pilot_validation', 'pipeline_synthetic')
        pred, records = asyncio.run(execute_pipeline(self.p, 'pilot_validation', EmptyWorkflow(), identity, 4))
        self.assertEqual(len(records), 5)
        self.assertEqual(len(pred), 10)  # k=0 only
        keys = expected_keys(self.p.select('pilot_validation'), ['llm_pipeline_linear_retrieval'])
        result = evaluate_rankings(pred, **self.p.private_truth(), expected_keys=keys, run_records=records)
        self.assertTrue(result[result.k > 0]['ndcg@10'].eq(0).all())
        # Completed cached runs require no additional calls.
        class NeverWorkflow:
            async def run(self, query):
                raise AssertionError('Should be cached')
        again, _ = asyncio.run(execute_pipeline(self.p, 'pilot_validation', NeverWorkflow(), identity, 0))
        pd.testing.assert_frame_equal(pred, again)
        failed_identity = run_identity(self.p, 'pilot_validation', 'pipeline_failed')
        with self.assertRaises(RuntimeError):
            asyncio.run(execute_pipeline(self.p, 'pilot_validation', NeverWorkflow(), failed_identity, 4))
        resumed, completed = asyncio.run(execute_pipeline(self.p, 'pilot_validation', EmptyWorkflow(), failed_identity, 4))
        self.assertEqual(len(completed), 5)

    def test_semantic_dict_validation_and_k0_fallback(self):
        neutral = pd.DataFrame({'movieId': [1], 'rating': [3.]})
        profile = validate_semantic_profile({'positive_query': None, 'negative_query': None, 'brief_rationale': ''}, neutral)
        self.assertIsNone(profile.positive_query)
        with self.assertRaises(ValueError):
            validate_semantic_profile({'positive_query': 'drama', 'brief_rationale': ''}, neutral)
        selected = self.p.select('pilot_validation').query('k == 0')
        profiles = {(str(r.scenario_id), 0): SemanticProfile(brief_rationale='').model_dump() for r in selected.itertuples()}
        pred = generate_semantic_rankings(self.p, selected, profiles, {}, np.eye(50), {'llm_gte_direct': {'params': {}}})
        bayesian, _ = aggregate_scores(self.p.train, self.p.ids)
        ranked, _ = top_k_ranking(bayesian, self.p.ids, pd.DataFrame(columns=['movieId']))
        self.assertEqual(pred.movieId.tolist(), ranked.tolist())

    def test_benchmark_profile_does_not_change_fast_profile(self):
        from movie_rag.resource_domain import load_resource_domain_catalog
        from movie_rag.resource_mcp_profiles import load_resource_mcp_profiles
        from movie_rag.resource_service import RetrievalService
        from movie_rag.resource_models import ResourceSearchHit
        profiles = load_resource_mcp_profiles(REPO/'configs/resource_mcp_profiles_v1.json')
        profile = profiles.get('movielens_coldstart_benchmark_v2')
        class Retriever:
            name = 'synthetic'
            def search(self, query, **kwargs):
                return [ResourceSearchHit(resource_id=i, rank=i, score=1/i, title=str(i), text='x')
                        for i in range(1, kwargs['top_k']+1)]
        service = RetrievalService(Retriever(), profiles=profile.build_search_profiles())
        response = service.search('q', profile_id='benchmark', limit=30, exact=True)
        self.assertEqual(len(response.results), 30)
        self.assertTrue(profile.exact)
        self.assertIsNone(profile.reranker)
        self.assertEqual(profiles.get('movielens_gte_fast_v2').build_search_profiles()['fast'].final_k, 10)
        domain = load_resource_domain_catalog(REPO/'configs/resource_domains_v1.json').get(profile.domain_id)
        with self.assertRaises(ValueError):
            domain.validate_native_filters({'include_genres': ['Drama']})


class TestTemperature(unittest.TestCase):
    def test_explicit_override_and_provider_default_for_both_agents(self):
        from movie_rag.resource_domain import load_resource_domain_catalog
        from movie_rag.resource_llm_agents import ResourcePlannerAgent, ResourceAnswerAgent
        from movie_rag.resource_models import ResourceSearchResponse, PipelineInfo
        domain = load_resource_domain_catalog(REPO/'configs/resource_domains_v1.json').get('movielens_coldstart_v2')
        response = ResourceSearchResponse(status='no_results', query='q', returned_count=0,
            pipeline=PipelineInfo(profile_id='benchmark', retriever='fake', candidate_k=30, final_k=30, rerank_enabled=False))
        class Provider:
            def __init__(self): self.calls = []
            async def generate_text(self, **kwargs):
                self.calls.append(kwargs)
                data = ({'semantic_query': 'drama', 'filters': {}} if kwargs['agent'] == 'resource_planner'
                        else {'status': 'no_results', 'answer': 'none', 'recommendations': []})
                return NS(text=json.dumps(data), trace={})
        for temperature in [None, .37]:
            provider = Provider()
            asyncio.run(ResourcePlannerAgent(provider, domain, temperature=temperature).run('query'))
            asyncio.run(ResourceAnswerAgent(provider, domain, temperature=temperature).run('query', response))
            for kwargs in provider.calls:
                if temperature is None:
                    self.assertNotIn('temperature', kwargs)
                else:
                    self.assertEqual(kwargs['temperature'], temperature)


class TestSyntheticWorkflow(unittest.TestCase):
    """Exercise the actual entrypoints using only tiny generated data and fake IO."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.p, self.f = fixture(self.temp.name)
        c = self.p.config.values
        c['mf'].update(ranks=[2], epochs=[1], regularization=[.1], fold_regularization=[1.])
        c['semantic'].update(mixed_negative_alpha=[0., .2], negative_only_alpha=[0., .35], semantic_weights=[0., 1.])
        cfg = self.p.config.root/c['llm']['config_path']; cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_bytes((REPO/c['llm']['config_path']).read_bytes())

        class Encoder:
            def __init__(self): self.calls = 0
            def encode_documents(self, texts):
                self.calls += 1
                vectors = np.zeros((len(texts), 768), dtype=np.float32)
                for i, text in enumerate(texts):
                    vectors[i, int(digest(text)[:8], 16) % 768] = 1.
                return vectors
        self.encoder = Encoder()
        class Client:
            def __init__(self): self.exists = False; self.points = []
            def get_collections(self):
                return NS(collections=[NS(name=c['embedding']['collection_name'])] if self.exists else [])
            def scroll(self, **kwargs): return self.points, None
        self.client = Client()
        def ensure(client, config, recreate):
            self.assertFalse(recreate)
            client.exists = True
        def upsert(client, config, resources, vectors, embedding_model):
            from movie_rag.resource_qdrant import resource_payload
            client.points = [NS(payload=resource_payload(resource, config, embedding_model)) for resource in resources]
        self.ensure = patch('movie_rag.resource_qdrant.ensure_collection', side_effect=ensure)
        self.upsert = patch('movie_rag.resource_qdrant.upsert_resources', side_effect=upsert)
        self.ensure.start(); self.upsert.start()
        self.addCleanup(self.ensure.stop); self.addCleanup(self.upsert.stop)
        prepare_embeddings(self.p, encoder=self.encoder, client=self.client)

    def test_embedding_cache_reuse_and_incompatible_collection_rejected(self):
        vectors, fingerprint = load_vectors(self.p)
        self.assertEqual(vectors.shape, (50, 768))
        prepare_embeddings(self.p, encoder=self.encoder, client=self.client)
        self.assertEqual(self.encoder.calls, 1)
        self.client.points[0].payload['metadata']['benchmark_embedding_fingerprint'] = 'old'
        with self.assertRaises(ValueError):
            prepare_embeddings(self.p, encoder=self.encoder, client=self.client)
        self.assertTrue(fingerprint)

    def test_full_validation_freezes_test_and_llm_runs_are_resumable(self):
        with self.assertRaises(FileNotFoundError):
            run_baselines(self.p, 'test')
        validation = run_baselines(self.p, 'full_validation')
        test = run_baselines(self.p, 'test')
        self.assertEqual(set(test['macro'].model), {'bayesian_mean', 'most_popular', 'mf_bias_only',
                         'mf_explicit_foldin', 'gte_direct', 'gte_prior'})
        self.assertEqual(set(read_reports(self.p, 'test')['mode']), {'test'})
        class Provider:
            def __init__(self): self.calls = 0
            async def generate_structured(self, **kwargs):
                self.calls += 1
                payload = json.loads(kwargs['user_prompt'])
                return NS(value=SemanticProfile(
                    positive_query='drama' if any(r['rating'] >= 4 for r in payload) else None,
                    negative_query='horror' if any(r['rating'] <= 2 for r in payload) else None,
                    brief_rationale='Revealed ratings only'), trace=None)
        provider = Provider()
        semantic_val = asyncio.run(run_semantic_llm(self.p, 'full_validation', provider=provider,
                                                   encoder=self.encoder, max_new_calls=20))
        calls = provider.calls
        again = asyncio.run(run_semantic_llm(self.p, 'full_validation', provider=provider,
                                           encoder=self.encoder, max_new_calls=0))
        self.assertEqual(provider.calls, calls)
        pd.testing.assert_frame_equal(semantic_val['macro'], again['macro'])
        semantic_test = asyncio.run(run_semantic_llm(self.p, 'test', provider=provider,
                                                    encoder=self.encoder, max_new_calls=20))
        self.assertTrue({'llm_gte_direct', 'llm_gte_prior'}.issubset(set(semantic_test['macro'].model)))
        self.assertFalse(read_traces(self.p, 'test').empty)
        # A corrupted baseline report cannot silently be reused for a comparison.
        with (test['directory']/'rankings.csv').open('a') as stream:
            stream.write('corruption\n')
        with self.assertRaises(ValueError):
            asyncio.run(run_semantic_llm(self.p, 'test', provider=provider, encoder=self.encoder, max_new_calls=0))


if __name__ == '__main__':
    unittest.main()
