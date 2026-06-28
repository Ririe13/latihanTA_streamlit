# tfrf_model.py

import numpy as np
from scipy.sparse import coo_matrix
from collections import defaultdict

class TFRFMulticlass:

    def __init__(self):
        self.rf_matrix   = None
        self.vocabulary  = []
        self.vocab_index = {}
        self.classes     = []
        self.class_index = {}

    def fit(self, documents, labels):
        self.classes     = sorted(list(set(labels)))
        self.class_index = {c: i for i, c in enumerate(self.classes)}

        vocab_set        = set(w for doc in documents for w in doc.split())
        self.vocabulary  = sorted(vocab_set)
        self.vocab_index = {w: i for i, w in enumerate(self.vocabulary)}

        n_terms   = len(self.vocabulary)
        n_classes = len(self.classes)

        a_mat = np.zeros((n_terms, n_classes), dtype=np.float32)
        b_mat = np.zeros((n_terms, n_classes), dtype=np.float32)

        for doc, label in zip(documents, labels):
            cls_idx = self.class_index[label]
            words   = set(doc.split())
            for word in words:
                if word in self.vocab_index:
                    t = self.vocab_index[word]
                    a_mat[t, cls_idx] += 1
                    b_mat[t, :]       += 1
                    b_mat[t, cls_idx] -= 1

        self.rf_matrix = np.log10(2 + (a_mat / np.maximum(1, b_mat)))
        return self

    def transform(self, documents):
        n_docs    = len(documents)
        n_terms   = len(self.vocabulary)
        n_classes = len(self.classes)
        n_cols    = n_terms * n_classes

        rows_list = []
        cols_list = []
        data_list = []

        for doc_idx, doc in enumerate(documents):
            tf = defaultdict(int)
            for word in doc.split():
                tf[word] += 1

            for word, freq in tf.items():
                if word in self.vocab_index:
                    t          = self.vocab_index[word]
                    row_scores = self.rf_matrix[t, :] * freq

                    nonzero_c_idx = np.nonzero(row_scores)[0]
                    if nonzero_c_idx.size == 0:
                        continue

                    cols = t * n_classes + nonzero_c_idx
                    vals = row_scores[nonzero_c_idx]

                    rows_list.append(np.full(nonzero_c_idx.size, doc_idx, dtype=np.int64))
                    cols_list.append(cols.astype(np.int64))
                    data_list.append(vals.astype(np.float32))

        if data_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            data = np.concatenate(data_list)
        else:
            rows = np.array([], dtype=np.int64)
            cols = np.array([], dtype=np.int64)
            data = np.array([], dtype=np.float32)

        X = coo_matrix((data, (rows, cols)), shape=(n_docs, n_cols), dtype=np.float32)
        return X.tocsr()