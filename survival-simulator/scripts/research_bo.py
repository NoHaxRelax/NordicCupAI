"""Small Gaussian-process Bayesian search for selected scalar policy settings.

Fixed RBF kernel, standardized objective and noisy expected improvement. This
is an intentionally bounded BO implementation, not an optimizer benchmark claim.
NumPy/SciPy are existing simulator dependencies; imports are deferred until use.
"""
import math


def encode(config, specs, getter):
    vector = []
    for spec in specs:
        value = getter(config, spec['path'])
        if spec['kind'] == 'bool':
            vector.append(float(value))
            continue
        low, high = spec['low'], spec['high']
        if value is None:
            vector.extend((0., 1., 0.))
            continue
        # Extra flags distinguish disabled sentinels and null from bound values.
        number = float(value)
        outside = float(number < low or number > high)
        clipped = min(high, max(low, number))
        if low > 0 and high / low > 20:
            scaled = (math.log(clipped)-math.log(low)) / max(1e-12, math.log(high)-math.log(low))
        else:
            scaled = (clipped-low) / max(1e-12, high-low)
        vector.extend((scaled, 0., outside))
    return vector


def choose(pool, observations, specs, getter, count):
    """Select a diverse batch; only completed same-study cases train the GP."""
    import numpy as np
    from scipy.special import ndtr

    if len(observations) < 5 or not specs:
        return pool[:count]
    x = np.asarray([encode(row['config'], specs, getter) for row in observations])
    y = np.asarray([row['summary']['rank'][0] for row in observations])
    z = np.asarray([encode(config, specs, getter) for config in pool])
    if len(z) == 0:
        return []
    y = (y-y.mean()) / max(float(y.std()), 1e-6)

    def kernel(a, b):
        distance = ((a[:, None, :]-b[None, :, :])**2).sum(axis=2)
        return np.exp(-distance / (2 * max(1., len(specs)/4)))

    k = kernel(x, x) + np.eye(len(x)) * .05
    cross = kernel(x, z)
    try:
        chol = np.linalg.cholesky(k)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        mean = cross.T @ alpha
        solved = np.linalg.solve(chol, cross)
        sigma = np.sqrt(np.maximum(1e-9, 1.-(solved**2).sum(axis=0)))
    except np.linalg.LinAlgError:
        return pool[:count]
    improvement = mean - y.max() - .01
    normalized = improvement / sigma
    acquisition = improvement * ndtr(normalized) + sigma * np.exp(-normalized**2/2) / math.sqrt(2*math.pi)
    selected, indices = [], []
    for _ in range(min(count, len(pool))):
        masked = acquisition.copy()
        masked[indices] = -math.inf
        index = int(np.argmax(masked))
        selected.append(pool[index])
        indices.append(index)
        distance = ((z-z[index])**2).sum(axis=1)
        acquisition *= 1.-np.exp(-distance / .2)
    return selected
