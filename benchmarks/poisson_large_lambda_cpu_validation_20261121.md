# Poisson large-rate rejection: independent CPU numerical check (2026-11-21)

This note supplements the regression tests and the benchmark functions in `benchmarks/random_benchmark.py` for [PR #41228](https://github.com/jax-ml/jax/pull/41228). It is **not** an official JAX CI, GPU, or production performance result.

## Environment and provenance

- Linux CPU, Python 3.13, installed JAX **0.9.0.1**, SciPy **1.17.0**.
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`.
- The acceptance loop was reconstructed from PR head `735b5b1ab8cae36f6a269efa31a75f8107bea67c`, comparing its stable `_poisson_log_pmf` formula with the pre-fix `-lam + k*log(lam) - gammaln(k+1)` expression. The local implementation uses public `jax.random.split` and `jax.random.uniform` and adapts the helper's private constant construction to the installed JAX version. Consequently this is **independent algorithm evidence**, not an exact-source checkout.
- Each condition uses **12 independent seeds, 8192 `float32`-rate Poisson samples per seed**; the random key and proposal generator are paired between implementations.

| Rate | Pre-fix mean observed variance / rate | Stable calculation variance / rate | Paired seeds with smaller absolute variance error |
| --- | ---: | ---: | ---: |
| `1e4` | 1.008256 | 1.007929 | 5/12 |
| `1e8` | **1.467037** | **1.007854** | **12/12** |

The main finding is the observed removal of large-rate variance inflation, with no detectable change in ordinary-rate variance at this sample size. The large-rate sample size is `12 * 8192 = 98,304`; the expected Poisson variance is `lam`.

For an independent checkout of **this PR's actual code**, the following local CPU smoke check exercises the public `jax.random.poisson` interface without relying on any benchmark-specific private helper. It can run without a GPU or external network after installing the project:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python - <<'PY'
import numpy as np
import jax
import jax.numpy as jnp

for rate in (1e4, 1e8):
    ratios = []
    for seed in range(12):
        samples = np.asarray(
            jax.random.poisson(
                jax.random.key(20261121 + seed),
                jnp.full((8192,), rate, dtype=jnp.float32),
                dtype=jnp.int32,
            )
        ).astype(np.float64)
        ratios.append(samples.var() / rate)
    print(f"rate={rate:g}, mean(var/rate)={np.mean(ratios):.6f}, "
          f"range=[{min(ratios):.6f}, {max(ratios):.6f}]")
PY
```

The existing `poisson_rejection_standard_rate_float32` and `poisson_rejection_large_rate_float32` microbenchmarks in `benchmarks/random_benchmark.py` provide the corresponding *actual sampler* latency measurements. The local reconstructed sampler had median 8192-sample CPU dispatch/barrier times of approximately **1.91 ms pre-fix vs 1.82 ms stable** at `1e4`, and **1.92 ms pre-fix vs 1.83 ms stable** at `1e8`. These are only five small timing batches and do **not** establish general performance gains or the absence of GPU regressions.

## Interpretation and validation scope

- The numerical contrast supports the reason for the stability rewrite and complements the PR's SciPy comparison and high-precision tests.
- The public-interface smoke command above is provided for **future independent reproduction**; its exact numbers on a full current JAX checkout have **not** been claimed as already run here.
- Statistical accuracy, compiler behavior and performance still depend on the full JAX CI and the maintainers' review. No hidden model, paid service, or external benchmark submission was involved.
