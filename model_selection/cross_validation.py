import numpy as np

def cross_validate(model, X, y, k=5, random_state=None):
    """Perform k-fold cross-validation."""
    if random_state:
        np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_sizes = np.full(k, len(X) // k)
    fold_sizes[:len(X) % k] += 1  # Distribute the remainder
    current = 0
    scores = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        current = stop

    return np.mean(scores), np.std(scores)



