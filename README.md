<h1 align='center'>Brutax</h1>

[![Continuous Integration](https://github.com/michael-0brien/brutax/actions/workflows/ci_build.yml/badge.svg)](https://github.com/michael-0brien/brutax/actions/workflows/ci_build.yml)

When all else fails, why not a brute-force search! Brutax is a JAX library for function optimization by brute-force grid search. Features include

- Highly-parallel loss function evaluations
- PyTree-valued grids
- Support for custom search behavior downstream
- Smooth integration with JAX function transformations: JIT, autodiff, vectorization, and scaling across GPU/TPUs

## Installation

```
pip install brutax
```

## Acknowledgements

- The design of `brutax` is heavily inspired from the JAX non-linear optimization library [`optimistix`](https://github.com/patrick-kidger/optimistix/tree/main).
