from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .cards import Card
from .enums import RANK_COUNT, SUIT_COUNT

BoolTensor = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CardTensorSpec:
    """Metadata describing the layout of a CardTensor.

    Attributes:
        plane_names: Names for each feature plane along axis 2,
            e.g. ("my_hand", "opponent_hand", "discard").
    """

    plane_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        for name in self.plane_names:
            if not re.match(r"^[a-z][a-z0-9_]*$", name):
                raise ValueError(
                    f"Plane name '{name}' must be snake_case (lowercase letters, numbers, underscores)"
                )

    @property
    def num_planes(self) -> int:
        return len(self.plane_names)


@dataclass(frozen=True, slots=True)
class CardTensor:
    """Immutable 4×14×n tensor over the 52-card deck + Jokers.

    Axis 0: suits, length 4.
    Axis 1: ranks, length 14 (0-12 standard, 13 Joker).
    Axis 2: feature planes, length n.
    """

    data: BoolTensor
    spec: CardTensorSpec

    def __post_init__(self) -> None:
        if self.data.shape != (SUIT_COUNT, RANK_COUNT, self.spec.num_planes):
            msg = (
                "CardTensor data shape must be (4, 14, n); "
                f"got {self.data.shape} for {self.spec.num_planes} planes"
            )
            raise ValueError(msg)
        if self.data.dtype != np.bool_:
            msg = f"CardTensor dtype must be bool, got {self.data.dtype}"
            raise TypeError(msg)
        # Enforce logical immutability
        self.data.setflags(write=False)

    @classmethod
    def empty(cls, plane_names: Sequence[str]) -> "CardTensor":
        spec = CardTensorSpec(tuple(plane_names))
        data = np.zeros((SUIT_COUNT, RANK_COUNT, spec.num_planes), dtype=bool)
        return cls(data=data, spec=spec)

    @classmethod
    def from_plane_card_map(
        cls,
        plane_to_cards: Mapping[str, Iterable[Card]],
    ) -> "CardTensor":
        """Build a CardTensor from a mapping of plane name → iterable of Cards."""
        plane_names = tuple(plane_to_cards.keys())
        tensor = cls.empty(plane_names)
        data = tensor.data.copy()
        for plane_index, plane_name in enumerate(plane_names):
            for card in plane_to_cards[plane_name]:
                idx = card.to_index()
                suit_index, rank_index = divmod(idx, RANK_COUNT)
                data[suit_index, rank_index, plane_index] = True
        return cls(data=data, spec=tensor.spec)

    def cards_in_plane(self, plane_name: str) -> Tuple[Card, ...]:
        """Return all Cards with True in the given plane."""
        try:
            plane_index = self.spec.plane_names.index(plane_name)
        except ValueError as exc:  # plane not present
            raise KeyError(f"Unknown plane name: {plane_name}") from exc
        slice_ = self.data[:, :, plane_index]
        cards: list[Card] = []
        for idx in range(SUIT_COUNT * RANK_COUNT):
            suit_index, rank_index = divmod(idx, RANK_COUNT)
            if slice_[suit_index, rank_index]:
                cards.append(Card.from_index(idx))
        return tuple(cards)

    def plane(self, plane_name: str) -> BoolTensor:
        """Return the 4×14 slice for a given plane name."""
        try:
            plane_index = self.spec.plane_names.index(plane_name)
        except ValueError as exc:
            raise KeyError(f"Unknown plane name: {plane_name}") from exc
        return self.data[:, :, plane_index]

    def flat_plane(self, plane_name: str) -> BoolTensor:
        """Return a flattened 56-element vector for a plane (row-major)."""
        return self.plane(plane_name).reshape(-1)

    @classmethod
    def from_flat_plane(
        cls,
        plane_name: str,
        flat: BoolTensor,
    ) -> "CardTensor":
        """Create a single-plane tensor from a 56-element flat boolean vector."""
        if flat.size != SUIT_COUNT * RANK_COUNT:
            msg = f"Flat plane must have size 56, got {flat.size}"
            raise ValueError(msg)
        spec = CardTensorSpec((plane_name,))
        data = flat.reshape(SUIT_COUNT, RANK_COUNT, 1)
        return cls(data=data, spec=spec)

    def validate_exclusive_planes(self, plane_names: Sequence[str]) -> None:
        """Ensure each card is present in at most one of the given planes.

        Raises:
            ValueError: If any card is present in more than one of the planes.
        """
        try:
            indices = [self.spec.plane_names.index(name) for name in plane_names]
        except ValueError as exc:
            raise KeyError(f"Unknown plane name in validation list: {exc}") from exc

        subset = self.data[:, :, indices]  # shape (4, 14, k)
        # Sum across the plane axis; any > 1 indicates a violation.
        counts = subset.sum(axis=2)  # still bool -> auto casts to int
        if np.any(counts > 1):
            raise ValueError("Card appears in more than one of the exclusive planes.")

    def validate_partition(self, plane_names: Sequence[str]) -> None:
        """Ensure planes form a complete partition of the deck (each card in exactly one).

        Raises:
            ValueError: If any card is not in exactly one plane.
        """
        try:
            indices = [self.spec.plane_names.index(name) for name in plane_names]
        except ValueError as exc:
            raise KeyError(f"Unknown plane name in validation list: {exc}") from exc

        subset = self.data[:, :, indices]
        counts = subset.sum(axis=2)

        # Check for overlaps (>1)
        if np.any(counts > 1):
            raise ValueError("At least one card appears in multiple planes.")

        # Check for omissions (0)
        if np.any(counts == 0):
            raise ValueError("At least one card is not present in any plane.")
