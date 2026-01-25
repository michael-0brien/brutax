from typing import TypeAlias, TypeVar

from jaxtyping import Array, Int, PyTree, Shaped


SearchSolution = TypeVar("SearchSolution")
SearchState = TypeVar("SearchState")
PyTreeGrid: TypeAlias = PyTree[Shaped[Array, "_ ..."] | None, " Y"]  # type: ignore
PyTreeGridPoint: TypeAlias = PyTree[Shaped[Array, "..."] | None, " Y"]  # type: ignore
PyTreeGridIndex: TypeAlias = PyTree[Int[Array, "..."] | None, "... Y"]  # type: ignore
