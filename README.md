<h1 align='center'>Brutax</h1>

[![Continuous Integration](https://github.com/michael-0brien/brutax/actions/workflows/ci_build.yml/badge.svg)](https://github.com/michael-0brien/brutax/actions/workflows/ci_build.yml)

When all else fails, why not a brute-force search! Brutax is a JAX library for function optimization by brute-force grid search. Features include:

- Highly-parallel function evaluations
- PyTree-valued grids
- Support for custom search behavior downstream
- Smooth integration with JAX function transformations: JIT, autodiff, vectorization, and scaling across GPU/TPUs

## Installation

```
pip install brutax
```

## Quick example

```python
import brutax
import jax.numpy as jnp

def fn(tree_grid_point, _):
    x, y = tree_grid_point
    return (x - 1.)**2 + (y - 2.)**2

# The `tree_grid` is the cartesian product of x and y grids
x_grid, y_grid = (jnp.arange(-5., 5., 1), jnp.arange(-5., 5., 0.1))
tree_grid = (x_grid, y_grid)
# Run grid search over (10 x 100) grid
sol = brutax.run_grid_search(
    fn, method=brutax.MinimumSearchMethod(), tree_grid=tree_grid, args=None
)
x_min, y_min = sol.value
assert jnp.isclose(x_min, 1.) and jnp.isclose(y_min, 2.)
```

## Acknowledgements

- The design of `brutax` is heavily inspired from the JAX non-linear optimization library [`optimistix`](https://github.com/patrick-kidger/optimistix/tree/main).
