"""Cold-start ranking models. No data loading, network access or evaluation truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-9 else np.zeros_like(vector)


def standardized(scores):
    scores = np.asarray(scores, dtype=float)
    return (scores - scores.mean()) / max(float(scores.std()), 1e-8)


def aggregate_scores(train, movie_ids, prior=15.0):
    if train.empty or prior <= 0:
        raise ValueError("Nonempty training and positive Bayesian prior required")
    stats = train.groupby('movieId').rating.agg(['sum', 'count']).reindex(movie_ids, fill_value=0)
    mean = float(train.rating.mean())
    return ((stats['sum'].to_numpy(float) + prior * mean) /
            (stats['count'].to_numpy(float) + prior), stats['count'].to_numpy(float))


def top_k_ranking(scores, movie_ids, observed, top_k=10):
    ids = np.asarray(movie_ids, dtype=np.int64)
    scores = np.asarray(scores, dtype=float).copy()
    if scores.shape != ids.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid ranking scores")
    if len(np.unique(ids)) != len(ids) or top_k <= 0:
        raise ValueError("Unique catalog IDs and positive cutoff required")
    shown = set(observed.movieId.astype(int))
    if not shown.issubset(set(ids)):
        raise ValueError("Observed item outside common catalog")
    scores[np.isin(ids, list(shown))] = -np.inf
    order = np.lexsort((ids, -scores))
    order = order[np.isfinite(scores[order])][:top_k]
    return ids[order], scores[order]


@dataclass
class ExplicitMF:
    movie_ids: object
    rank: int = 24
    epochs: int = 18
    lr: float = .008
    reg: float = .06
    seed: int = 20261021

    def fit(self, frame):
        if frame.empty or self.rank < 0 or self.epochs < 0:
            raise ValueError("Invalid MF training data or parameters")
        self.movie_ids = np.asarray(self.movie_ids, dtype=np.int64)
        self.index = {int(mid): i for i, mid in enumerate(self.movie_ids)}
        if len(self.index) != len(self.movie_ids):
            raise ValueError("Duplicate catalog IDs")
        if not set(frame.movieId).issubset(self.index):
            raise ValueError("Training item outside catalog")
        users = np.sort(frame.userId.unique())
        lookup = {int(uid): i for i, uid in enumerate(users)}
        u = np.array([lookup[int(uid)] for uid in frame.userId])
        it = np.array([self.index[int(mid)] for mid in frame.movieId])
        ratings = frame.rating.to_numpy(float)
        self.mu = float(ratings.mean())
        self.user_bias = np.zeros(len(users))
        self.item_bias = np.zeros(len(self.movie_ids))
        self.item_counts = np.bincount(it, minlength=len(self.movie_ids))
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, .065, (len(users), self.rank))
        self.Q = rng.normal(0, .065, (len(self.movie_ids), self.rank))
        # Unknown items have a deterministic global-mean fallback, including in fold-in.
        self.Q[self.item_counts == 0] = 0
        self.loss_history = []
        for epoch in range(self.epochs):
            for j in rng.permutation(len(ratings)):
                a, b = u[j], it[j]
                pa, qb = self.P[a].copy(), self.Q[b].copy()
                error = ratings[j] - (self.mu + self.user_bias[a] + self.item_bias[b] + pa @ qb)
                self.user_bias[a] += self.lr * (error - self.reg * self.user_bias[a])
                self.item_bias[b] += self.lr * (error - self.reg * self.item_bias[b])
                self.P[a] += self.lr * (error * qb - self.reg * pa)
                self.Q[b] += self.lr * (error * pa - self.reg * qb)
            if epoch == 0 or epoch + 1 == self.epochs or (epoch + 1) % 5 == 0:
                pred = self.mu + self.user_bias[u] + self.item_bias[it] + (self.P[u] * self.Q[it]).sum(1)
                self.loss_history.append({'epoch': epoch + 1, 'train_rmse': float(np.sqrt(np.mean((ratings-pred)**2)))})
        return self

    def fold_in(self, observed, fold_reg=5.0):
        if fold_reg <= 0:
            raise ValueError("Positive fold-in regularization required")
        if observed.empty:
            return 0.0, np.zeros(self.rank)
        ix = np.array([self.index[int(mid)] for mid in observed.movieId])
        design = np.column_stack([np.ones(len(ix)), self.Q[ix]])
        target = observed.rating.to_numpy(float) - self.mu - self.item_bias[ix]
        penalty = np.eye(self.rank + 1) * fold_reg
        penalty[0, 0] = max(.1, fold_reg * .25)
        theta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return float(theta[0]), theta[1:]

    def score(self, observed, fold_reg=5.0):
        bias, factors = self.fold_in(observed, fold_reg)
        return self.mu + self.item_bias + bias + self.Q @ factors


def combine_semantic_scores(positive, negative, bayesian,
                            mixed_alpha=.5, negative_only_alpha=.35, semantic_weight=1.0):
    """Mix the prior only in the positive branch; retain the legacy negative fallback."""
    if not 0 <= semantic_weight <= 1 or min(mixed_alpha, negative_only_alpha) < 0:
        raise ValueError("Invalid semantic scoring parameters")
    if positive is not None:
        semantic = positive
        if negative is not None:
            semantic = semantic - mixed_alpha * negative
        if semantic_weight == 1:
            return semantic  # pure GTE remains an explicit baseline
        return ((1-semantic_weight) * standardized(bayesian) +
                semantic_weight * standardized(semantic))
    if negative is not None:
        return standardized(bayesian) - negative_only_alpha * negative * 3
    return np.asarray(bayesian, dtype=float).copy()


def semantic_scores(vectors, positive_vector, negative_vector, bayesian, **params):
    return combine_semantic_scores(
        vectors @ unit(positive_vector) if positive_vector is not None else None,
        vectors @ unit(negative_vector) if negative_vector is not None else None, bayesian, **params)


def gte_affinities(observed, movie_ids, vectors, positive_min=4., negative_max=2.):
    index = {int(mid): i for i, mid in enumerate(movie_ids)}
    def centroid(frame, positive):
        if frame.empty:
            return None
        weights = (frame.rating.to_numpy(float)-3.5 if positive else 2.5-frame.rating.to_numpy(float))
        return np.average(vectors[[index[int(mid)] for mid in frame.movieId]], axis=0,
                          weights=np.maximum(weights, .1))
    positive = centroid(observed[observed.rating >= positive_min], True)
    negative = centroid(observed[observed.rating <= negative_max], False)
    return (vectors @ unit(positive) if positive is not None else None,
            vectors @ unit(negative) if negative is not None else None)


def gte_scores(observed, movie_ids, vectors, bayesian, positive_min=4., negative_max=2., **params):
    return combine_semantic_scores(*gte_affinities(observed, movie_ids, vectors, positive_min, negative_max),
                                   bayesian, **params)
