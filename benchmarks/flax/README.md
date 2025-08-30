# Tracing and lowering benchmarks for Flax examples

See Flax
[documentation](https://flax.readthedocs.io/en/latest/examples/index.html) on
their examples.

## Getting started
bash
```
pip install -r benchmarks/flax/requirements.txt

# Benchmark trace and lower timing for all workloads.
python tracing_benchmark.py

# Profile a single example.
python tracing_benchmark.py --example=wmt
```