import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class VectorizedBM25:
    """
    High-performance vectorized BM25 Okapi model implemented via sparse matrix multiplication.
    Equivalent to Robertson-Spärck Jones Okapi BM25 scoring with k1 and b parameters.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.vectorizer = None
        self.idf = None
        self.avgdl = 1.0
        self.M_T = None
        self.Q = None

    def fit(self, texts):
        self.vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
            dtype=np.float32,
        )
        X_cnt = self.vectorizer.fit_transform(texts)
        doc_lens = np.array(X_cnt.sum(axis=1), dtype=np.float32).flatten()
        self.avgdl = float(np.mean(doc_lens)) if len(doc_lens) > 0 else 1.0
        if self.avgdl == 0:
            self.avgdl = 1.0

        N = len(texts)
        df = np.bincount(X_cnt.indices, minlength=X_cnt.shape[1])
        # Standard Okapi BM25 IDF: ln((N - df + 0.5) / (df + 0.5) + 1.0)
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        # Precompute document term weights (M)
        X_coo = X_cnt.tocoo()
        len_norm = 1.0 - self.b + self.b * (doc_lens[X_coo.row] / self.avgdl)
        denom = X_coo.data + self.k1 * len_norm
        m_data = (X_coo.data * (self.k1 + 1.0)) / denom
        M = csr_matrix((m_data, (X_coo.row, X_coo.col)), shape=X_cnt.shape, dtype=np.float32)
        self.M_T = M.T.tocsr()

        # Precompute query term weights (Q)
        q_data = X_coo.data * self.idf[X_coo.col]
        self.Q = csr_matrix((q_data, (X_coo.row, X_coo.col)), shape=X_cnt.shape, dtype=np.float32)
        return self


def build_tfidf(texts, max_features=None):
    v = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        smooth_idf=True,
        sublinear_tf=True,
        norm="l2",
        max_features=max_features,
        dtype=np.float32,
    )
    return v.fit_transform(texts), v


def build_bm25(texts, k1=1.5, b=0.75):
    return VectorizedBM25(k1=k1, b=b).fit(texts)


def retrieve_tfidf(matrix, uids, k=5, batch_size=500):
    """
    Retrieve top-k similar patients for each patient using cosine similarity on TF-IDF.
    Vectorized and batched for high memory efficiency and speed.
    """
    result = {}
    N = len(uids)
    matrix = matrix.tocsr()
    matrix_T = matrix.T.tocsr()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        scores = matrix[start:end].dot(matrix_T).toarray()
        batch_len = end - start

        # Exclude self
        scores[np.arange(batch_len), np.arange(start, end)] = -np.inf

        count = min(k, max(0, N - 1))
        if count > 0:
            top_part = np.argpartition(scores, -count, axis=1)[:, -count:]
            row_idx = np.arange(batch_len)[:, None]
            part_scores = scores[row_idx, top_part]
            sorted_order = np.argsort(-part_scores, axis=1)
            final_top = top_part[row_idx, sorted_order]

            for row, global_idx in enumerate(range(start, end)):
                result[uids[global_idx]] = [uids[j] for j in final_top[row]]
        else:
            for global_idx in range(start, end):
                result[uids[global_idx]] = []

        if N >= 10000 and (end % 25000 < batch_size or end == N):
            print(f"      TF-IDF processed {end:,}/{N:,} patients ({end / N * 100:.1f}%)")

    return result


def retrieve_bm25(model, texts, uids, k=5, batch_size=500):
    """
    Retrieve top-k similar patients for each patient using BM25 scoring.
    Uses sparse matrix dot product: Q.dot(M.T).
    """
    if not hasattr(model, "M_T") or model.M_T is None:
        model = build_bm25(texts)

    result = {}
    N = len(uids)
    Q = model.Q
    M_T = model.M_T

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        scores = Q[start:end].dot(M_T).toarray()
        batch_len = end - start

        # Exclude self
        scores[np.arange(batch_len), np.arange(start, end)] = -np.inf

        count = min(k, max(0, N - 1))
        if count > 0:
            top_part = np.argpartition(scores, -count, axis=1)[:, -count:]
            row_idx = np.arange(batch_len)[:, None]
            part_scores = scores[row_idx, top_part]
            sorted_order = np.argsort(-part_scores, axis=1)
            final_top = top_part[row_idx, sorted_order]

            for row, global_idx in enumerate(range(start, end)):
                result[uids[global_idx]] = [uids[j] for j in final_top[row]]
        else:
            for global_idx in range(start, end):
                result[uids[global_idx]] = []

        if N >= 10000 and (end % 25000 < batch_size or end == N):
            print(f"      BM25 processed {end:,}/{N:,} patients ({end / N * 100:.1f}%)")

    return result


