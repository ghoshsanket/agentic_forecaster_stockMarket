"""Extract attention weights as human-interpretable evidence."""

from __future__ import annotations

import numpy as np
import torch


def extract_attention_evidence(
    model, X: np.ndarray, dates: np.ndarray, top_k: int = 5
) -> list[dict]:
    """Return the top-k most-attended days for each sample.

    Parameters
    ----------
    model : AttentionLSTM (must support ``return_attention=True``).
    X : input sequences, shape ``(n, T, F)``.
    dates : array of length ``n`` with the prediction date for each sample.
    top_k : number of top-attended days to return per sample.
    """
    model.eval()
    with torch.no_grad():
        _, weights = model(torch.tensor(X, dtype=torch.float32), return_attention=True)
    w = weights.cpu().numpy()  # (n, T)
    seq_len = X.shape[1]
    results = []
    for i in range(len(X)):
        top_idx = np.argsort(w[i])[::-1][:top_k]
        results.append(
            {
                "prediction_date": str(dates[i]),
                "top_days": [
                    {
                        "day_offset": int(seq_len - 1 - idx),
                        "attention_weight": float(w[i][idx]),
                    }
                    for idx in top_idx
                ],
            }
        )
    return results
