# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to evaluate the bias of jax.random samplers against analytical expectations.

Usage:
  python evaluate_sampler_bias.py --sampler=gamma --a=2.0 --method=exact --num_samples=1000000
  python evaluate_sampler_bias.py --sampler=normal --num_samples=1000000
  python evaluate_sampler_bias.py --sampler=uniform --minval=-1.0 --maxval=1.0
  python evaluate_sampler_bias.py --sampler=beta --a=0.5 --b=0.5
"""

import argparse
import dataclasses
from typing import Any

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats

try:
  import tabulate  # pyrefly: ignore[missing-import,missing-source-for-stubs]  # pytype: disable=import-error
except ImportError:
  tabulate = None


@dataclasses.dataclass(frozen=True)
class QuantileResult:
  quantile: float
  empirical: float
  analytical: float
  abs_error: float
  rel_error: float
  std_error: float
  z_score: float


@dataclasses.dataclass(frozen=True)
class EvaluationResult:
  sampler_name: str
  kwargs: dict[str, Any]
  num_samples: int
  seed: int
  dtype: str
  quantiles: list[float]
  results: list[QuantileResult]
  max_abs_error: float
  max_rel_error: float
  max_z_score: float


def parse_value_string(val_str: str) -> Any:
  """Converts string argument into float, int, bool, or keeps as string."""
  if val_str.lower() in ("true", "yes"):
    return True
  if val_str.lower() in ("false", "no"):
    return False
  try:
    return int(val_str)
  except ValueError:
    pass
  try:
    return float(val_str)
  except ValueError:
    pass
  return val_str


def parse_extra_kwargs(args_list: list[str]) -> dict[str, Any]:
  """Parses extra CLI arguments into a kwargs dictionary."""
  kwargs = {}
  i = 0
  while i < len(args_list):
    arg = args_list[i]
    if arg.startswith("--"):
      key_val = arg[2:].split("=", 1)
      key = key_val[0]
      if len(key_val) == 2:
        val_str = key_val[1]
      elif i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
        val_str = args_list[i + 1]
        i += 1
      else:
        val_str = "True"
      kwargs[key] = parse_value_string(val_str)
    i += 1
  return kwargs


def get_scipy_distribution(sampler_name: str, kwargs: dict[str, Any]) -> Any:
  """Returns the corresponding scipy.stats distribution for a given sampler."""
  sname = sampler_name.lower()

  if (
      sname in ("bernoulli", "poisson", "geometric", "binomial")
      or isinstance(getattr(scipy.stats, sname, None), scipy.stats.rv_discrete)
  ):
    raise ValueError(
        f"Discrete distributions like '{sampler_name}' are not supported for analytical quantile evaluation."
    )

  if sname == "gamma":
    a = kwargs.get("a", 1.0)
    return scipy.stats.gamma(a=a, scale=1.0)
  elif sname in ("normal", "standard_normal"):
    loc = kwargs.get("loc", 0.0)
    scale = kwargs.get("scale", 1.0)
    return scipy.stats.norm(loc=loc, scale=scale)
  elif sname == "uniform":
    minval = kwargs.get("minval", 0.0)
    maxval = kwargs.get("maxval", 1.0)
    return scipy.stats.uniform(loc=minval, scale=maxval - minval)
  elif sname == "beta":
    a = kwargs.get("a", 1.0)
    b = kwargs.get("b", 1.0)
    return scipy.stats.beta(a=a, b=b)
  elif sname == "exponential":
    return scipy.stats.expon(scale=1.0)
  elif sname == "cauchy":
    return scipy.stats.cauchy(loc=0.0, scale=1.0)
  elif sname == "laplace":
    return scipy.stats.laplace(loc=0.0, scale=1.0)
  elif sname == "logistic":
    return scipy.stats.logistic(loc=0.0, scale=1.0)
  elif sname == "pareto":
    b = kwargs.get("b", 1.0)
    return scipy.stats.pareto(b=b, scale=1.0)
  elif sname in ("chisquare", "chi2"):
    df = kwargs.get("df", 1.0)
    return scipy.stats.chi2(df=df)
  elif sname == "gumbel":
    return scipy.stats.gumbel_r(loc=0.0, scale=1.0)
  elif sname == "t":
    df = kwargs.get("df", 1.0)
    return scipy.stats.t(df=df)
  elif sname == "weibull_min":
    scale = kwargs.get("scale", 1.0)
    concentration = kwargs.get("concentration", 1.0)
    return scipy.stats.weibull_min(c=concentration, scale=scale)
  elif sname == "triangular":
    left = kwargs.get("left", -1.0)
    mode = kwargs.get("mode", 0.0)
    right = kwargs.get("right", 1.0)
    c = (mode - left) / (right - left)
    return scipy.stats.triang(c=c, loc=left, scale=right - left)
  elif sname == "lognormal":
    sigma = kwargs.get("sigma", 1.0)
    return scipy.stats.lognorm(s=sigma, scale=1.0)
  elif sname == "rayleigh":
    scale = kwargs.get("scale", 1.0)
    return scipy.stats.rayleigh(scale=scale)
  elif sname == "wald":
    mean = kwargs.get("mean", 1.0)
    scale = kwargs.get("scale", 1.0)
    return scipy.stats.invgauss(mu=mean / scale, scale=scale)
  elif sname == "maxwell":
    scale = kwargs.get("scale", 1.0)
    return scipy.stats.maxwell(scale=scale)
  elif hasattr(scipy.stats, sname):
    dist_cls = getattr(scipy.stats, sname)
    dist = dist_cls(**kwargs)
    if isinstance(getattr(dist, "dist", dist), scipy.stats.rv_discrete):
      raise ValueError(
          f"Discrete distributions like '{sampler_name}' are not supported for analytical quantile evaluation."
      )
    return dist
  else:
    raise ValueError(
        f"Unknown or unsupported distribution for analytical quantiles: {sampler_name}"
    )


def evaluate_bias(
    sampler_name: str,
    kwargs: dict[str, Any],
    num_samples: int,
    seed: int,
    dtype_str: str,
    quantiles: list[float],
) -> EvaluationResult:
  """Evaluates the sampler bias across specified quantiles."""
  if not hasattr(jax.random, sampler_name):
    raise ValueError(f"jax.random has no sampler named '{sampler_name}'")

  sampler_fn = getattr(jax.random, sampler_name)

  dtype = getattr(jnp, dtype_str, None)
  if dtype is None:
    dtype = np.dtype(dtype_str)

  key = jax.random.PRNGKey(seed)

  jax_kwargs = dict(kwargs)

  # Generate samples
  try:
    samples = sampler_fn(key, shape=(num_samples,), dtype=dtype, **jax_kwargs)
  except TypeError:
    try:
      samples = sampler_fn(key, shape=(num_samples,), **jax_kwargs)
    except TypeError:
      samples = sampler_fn(key, **jax_kwargs)

  samples_np = np.asarray(samples, dtype=np.float64)

  scipy_dist = get_scipy_distribution(sampler_name, jax_kwargs)

  empirical_quantiles = np.quantile(samples_np, quantiles)
  analytical_quantiles = scipy_dist.ppf(quantiles)

  abs_errors = empirical_quantiles - analytical_quantiles

  rel_errors = []
  for _, ana, err in zip(empirical_quantiles, analytical_quantiles, abs_errors):
    if ana != 0:
      rel_errors.append(err / ana)
    else:
      rel_errors.append(0.0 if err == 0 else np.nan)

  std_errors = []
  z_scores = []
  pdf_fn = getattr(scipy_dist, "pdf", None)

  for q, ana, err in zip(quantiles, analytical_quantiles, abs_errors):
    try:
      if pdf_fn is not None:
        dens = pdf_fn(ana)
        if dens > 0:
          se = np.sqrt(q * (1.0 - q)) / (np.sqrt(num_samples) * dens)
          z = err / se
        else:
          se = np.nan
          z = np.nan
      else:
        se = np.nan
        z = np.nan
    except Exception:  # pylint: disable=broad-exception-caught
      se = np.nan
      z = np.nan
    std_errors.append(se)
    z_scores.append(z)

  results = []
  for q, emp, ana, abs_err, rel_err, se, z in zip(
      quantiles,
      empirical_quantiles,
      analytical_quantiles,
      abs_errors,
      rel_errors,
      std_errors,
      z_scores,
  ):
    results.append(
        QuantileResult(
            quantile=q,
            empirical=emp,
            analytical=ana,
            abs_error=abs_err,
            rel_error=rel_err,
            std_error=se,
            z_score=z,
        )
    )

  return EvaluationResult(
      sampler_name=sampler_name,
      kwargs=jax_kwargs,
      num_samples=num_samples,
      seed=seed,
      dtype=dtype_str,
      quantiles=quantiles,
      results=results,
      max_abs_error=float(np.max(np.abs(abs_errors))),
      max_rel_error=float(
          np.nanmax(np.abs([r for r in rel_errors if not np.isnan(r)]))
          if any(not np.isnan(r) for r in rel_errors)
          else 0.0
      ),
      max_z_score=float(
          np.nanmax(np.abs([z for z in z_scores if not np.isnan(z)]))
          if any(not np.isnan(z) for z in z_scores)
          else 0.0
      ),
  )


def format_table_output(data: EvaluationResult) -> str:
  """Formats evaluation results into a human-readable table."""
  headers = [
      "Quantile",
      "Empirical",
      "Analytical",
      "Abs Error",
      "Rel Error",
      "Std Error",
      "Z-Score",
  ]
  rows = []
  for r in data.results:
    rel_err_str = (
        f"{r.rel_error * 100:+.4f}%" if not np.isnan(r.rel_error) else "N/A"
    )
    se_str = f"{r.std_error:.4e}" if not np.isnan(r.std_error) else "N/A"
    z_str = f"{r.z_score:+.2f}" if not np.isnan(r.z_score) else "N/A"
    rows.append([
        f"{r.quantile:.4g}",
        f"{r.empirical:.6g}",
        f"{r.analytical:.6g}",
        f"{r.abs_error:+.4e}",
        rel_err_str,
        se_str,
        z_str,
    ])

  header_text = (
      f"================================================================================\n"
      f"Sampler Bias Evaluation: jax.random.{data.sampler_name}\n"
      f"  Arguments: {data.kwargs}\n"
      f"  Num Samples: {data.num_samples:,} | Seed: {data.seed} | Dtype: {data.dtype}\n"
      f"================================================================================\n"
  )

  if tabulate is not None:
    table_str = tabulate.tabulate(rows, headers=headers, tablefmt="grid")
  else:
    col_widths = [len(h) for h in headers]
    for row in rows:
      for i, val in enumerate(row):
        col_widths[i] = max(col_widths[i], len(str(val)))
    header_line = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    row_lines = [
        " | ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
        for row in rows
    ]
    table_str = "\n".join([header_line, sep_line] + row_lines)

  summary_text = (
      f"\nSummary:\n"
      f"  Max Absolute Error: {data.max_abs_error:.4e}\n"
      f"  Max Relative Error: {data.max_rel_error * 100:.4f}%\n"
      f"  Max Z-Score:        {data.max_z_score:.2f}\n"
  )
  return header_text + table_str + summary_text


def main(argv: list[str]) -> None:
  parser = argparse.ArgumentParser(
      description="Evaluate bias of jax.random samplers against analytical expectations."
  )
  parser.add_argument(
      "--sampler",
      type=str,
      required=True,
      help="Name of jax.random sampler (e.g. gamma, normal, uniform, beta)",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=1000000,
      help="Number of samples to generate (default: 1000000)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="PRNGKey seed (default: 0)",
  )
  parser.add_argument(
      "--dtype",
      type=str,
      default="float32",
      help="Data type for generated samples (e.g. float32, float64)",
  )
  parser.add_argument(
      "--quantiles",
      type=str,
      default="0.001,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999",
      help="Comma-separated list of quantiles to evaluate",
  )

  args, unknown_args = parser.parse_known_args(argv[1:])

  sampler_name = args.sampler

  extra_kwargs = parse_extra_kwargs(unknown_args)
  quantile_list = [float(q.strip()) for q in args.quantiles.split(",") if q.strip()]

  data = evaluate_bias(
      sampler_name=sampler_name,
      kwargs=extra_kwargs,
      num_samples=args.num_samples,
      seed=args.seed,
      dtype_str=args.dtype,
      quantiles=quantile_list,
  )

  output = format_table_output(data)
  print(output)


if __name__ == "__main__":
  app.run(main, flags_parser=lambda args: flags.FLAGS(args, known_only=True))
