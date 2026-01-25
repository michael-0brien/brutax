from typing import TypeAlias, TypeVar

from jaxtyping import Array, Int, PyTree, Shaped


Y = TypeVar("Y")

PyTreeGrid: TypeAlias = PyTree[Shaped[Array, "_ ..."] | None, " Y"]
PyTreeGridPoint: TypeAlias = PyTree[Shaped[Array, "..."] | None, " Y"]
PyTreeGridIndex: TypeAlias = PyTree[Int[Array, "..."] | None, "... Y"]  # type: ignore
