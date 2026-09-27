import numpy as np


def predict_ensemble(models, images, mode='hard', weights=None, batch_size=64):
    """Votação majoritária; empate por média das probabilidades dos modelos empatados."""
    if not models or mode not in {'hard', 'soft'}:
        raise ValueError('Forneça modelos e escolha hard ou soft.')
    probabilities = np.stack([m.predict(images, batch_size=batch_size, verbose=0)
                              for m in models])
    if len({p.shape for p in probabilities}) != 1:
        raise ValueError('Os modelos precisam ter as mesmas classes, na mesma ordem.')
    if mode == 'soft':
        weights = np.ones(len(models)) if weights is None else np.asarray(weights, dtype=float)
        if weights.shape != (len(models),) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError('Informe um peso não negativo por modelo, com soma positiva.')
        combined = np.average(probabilities, axis=0, weights=weights)
        return combined.argmax(axis=1), combined
    votes = probabilities.argmax(axis=2)
    predictions = []
    for column in range(votes.shape[1]):
        counts = np.bincount(votes[:, column], minlength=probabilities.shape[2])
        tied = np.flatnonzero(counts == counts.max())
        if len(tied) == 1:
            predictions.append(int(tied[0]))
        else:
            supporting = np.isin(votes[:, column], tied)
            mean_prob = probabilities[supporting, column].mean(axis=0)
            predictions.append(int(tied[np.argmax(mean_prob[tied])]))
    return np.asarray(predictions), probabilities
