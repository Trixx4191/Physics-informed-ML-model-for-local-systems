import numpy as np
from data import generator


def test_rivers_defined():
    assert isinstance(generator.RIVERS, dict)
    assert "volta" in generator.RIVERS


def test_solve_small_grid():
    x, t, h, u, cfg = generator.solve_saint_venant("volta", Nx=12, Nt=24, T=3600)
    assert x.shape[0] == 12
    assert t.shape[0] >= 24
    assert h.shape[0] == t.shape[0]
    assert h.shape[1] == x.shape[0]
    # check values are finite
    assert np.isfinite(h).all()
    assert np.isfinite(u).all()


def test_generate_sparse_observations():
    x, t, h, u, _ = generator.solve_saint_venant("volta", Nx=12, Nt=24, T=3600)
    train, full, stats = generator.generate_sparse_observations(x, t, h, u, fraction=0.1)
    assert "xt" in train and "h" in train and "u" in train
    assert isinstance(stats, dict)
