"""Regression tests for a `jax.shard_map` carry-type mismatch in
`brutax.run_grid_search`.

Run this file on its own, e.g. `python -m pytest
tests/test_shard_map_manual_axes.py`, not bundled into a `pytest tests/` run
with other test files. `XLA_FLAGS` (below) must be set before jax's backend
initializes, which happens lazily on first device use and only once per
process; if another test file uses jax first, this module silently runs on
one simulated CPU device instead of the two it needs.
"""

import os


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import brutax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.sharding as jshard  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


N_DEVICES = 2


@pytest.fixture(autouse=True, scope="module")
def _require_two_devices():
    if jax.local_device_count() != N_DEVICES:
        pytest.skip(
            f"Expected {N_DEVICES} CPU devices via XLA_FLAGS, got "
            f"{jax.local_device_count()}. Run this file in its own process, "
            "not bundled with other tests that import jax first."
        )


def objective(grid_point, images):
    """A per-grid-point function whose output actually depends on the
    per-device (sharded) `images`, not just the grid point. This matters
    because the bug below only shows up when `fn`'s output genuinely differs
    from one shard to the next (i.e. varies over the sharded axis); an
    `objective` that ignored `images` would pass by accident.
    """
    (phase,) = grid_point
    return jnp.sum((jnp.cos(phase) - images) ** 2, axis=-1)


def test_run_grid_search_hangs_or_errors_without_the_fix_disabled():
    """Confirms the underlying `jax.shard_map` behavior that
    `MinimumSearchMethod.init`/`run_grid_search` work around still exists,
    by reproducing it directly: build a `_MinimumState` the way
    `MinimumSearchMethod.init` used to (from `f_struct.shape`/`.dtype`
    alone), then run it through the same `jax.lax.fori_loop` `run_grid_search`
    uses. `jax.lax.fori_loop` requires its carry to have the same type on
    every iteration; a state built from bare shape/dtype starts out not
    varying over the sharded axis, but `update()`'s real output does vary,
    so the loop should raise. If this test ever stops raising, that
    underlying jax behavior has changed, and the workaround in
    `MinimumSearchMethod.init`/`run_grid_search` should be re-evaluated.
    """
    mesh = jax.make_mesh(
        axis_sizes=(N_DEVICES,),
        axis_names=("batch_dim",),
        axis_types=(jshard.AxisType.Explicit,),
    )
    images = jax.device_put(
        jnp.ones((N_DEVICES, 5)),
        jshard.NamedSharding(mesh, jshard.PartitionSpec("batch_dim")),
    )
    grid = (jnp.linspace(0.0, 2 * jnp.pi, 8),)

    def per_device(images):
        method = brutax.MinimumSearchMethod()
        test_point = (grid[0][0],)
        f_struct = jax.eval_shape(objective, test_point, images)
        # Construct `_MinimumState` directly from `f_struct.shape`/`.dtype`,
        # bypassing `method.init` (which no longer seeds state this way).
        from brutax._method import _MinimumState

        state = _MinimumState(
            minimum_eval=jnp.full(f_struct.shape, jnp.inf, dtype=float),
            best_raveled_index=jnp.full(f_struct.shape, 0, dtype=int),
            current_eval=None,
        )

        def body(i, state):
            return method.update(objective, (grid[0][i],), images, state, i)

        with pytest.raises(TypeError, match="manual axis types do not match"):
            jax.lax.fori_loop(0, 8, body, state)
        return jnp.array(0.0)

    jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=jshard.PartitionSpec("batch_dim"),
        out_specs=jshard.PartitionSpec(),
    )(images)


def test_run_grid_search_under_shard_map_matches_unsharded_reference():
    """`brutax.run_grid_search`, called from inside `jax.shard_map` with one
    device's worth of data per shard, must complete (not raise, not hang)
    and produce the same result as running the identical search per device
    without any sharding at all.
    """
    grid = (jnp.linspace(0.0, 2 * jnp.pi, 33),)
    images_per_device = [jnp.full((5,), 0.3), jnp.full((5,), 0.9)]

    # Reference: run the exact same search per device, no sharding involved.
    method = brutax.MinimumSearchMethod()
    reference = [
        brutax.run_grid_search(objective, method, grid, images)
        for images in images_per_device
    ]

    mesh = jax.make_mesh(
        axis_sizes=(N_DEVICES,),
        axis_names=("batch_dim",),
        axis_types=(jshard.AxisType.Explicit,),
    )
    images_sharded = jax.device_put(
        jnp.stack(images_per_device),
        jshard.NamedSharding(mesh, jshard.PartitionSpec("batch_dim")),
    )

    def per_device(images):
        # Return just the array field under test, not the whole
        # `_MinimumSolution` -- `grid_shape` is a static, mesh-invariant
        # tuple, so it needs its own (replicated) out_specs entry, which
        # would otherwise turn this into a test of shard_map's pytree
        # out_specs handling rather than of brutax's own fix.
        method = brutax.MinimumSearchMethod()
        return brutax.run_grid_search(objective, method, grid, images).state.minimum_eval

    minimum_eval = jax.shard_map(
        per_device,
        mesh=mesh,
        in_specs=jshard.PartitionSpec("batch_dim"),
        out_specs=jshard.PartitionSpec("batch_dim"),
    )(images_sharded)
    minimum_eval = np.asarray(jax.device_get(minimum_eval))

    for device_index in range(N_DEVICES):
        np.testing.assert_allclose(
            minimum_eval[device_index],
            reference[device_index].state.minimum_eval,
            atol=1e-6,
        )
