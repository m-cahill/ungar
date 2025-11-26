Here we go: **M01 = “Core Card Physics”** — define the canonical 52-card universe and the 4×13×n tensor substrate, while keeping the M00 harness totally green.

---

## M01 – Core Card Tensor & Card Domain Primitives

### 🎯 High-Level Goals

By the end of M01, the `ungar` repo should have:

1. A **canonical card domain**:

   * `Suit` / `Rank` enums
   * An immutable `Card` value object
   * Deterministic mapping between `Card` ↔ integer index ↔ `(suit, rank)` grid
2. A **4×13×n tensor representation** (`CardTensor`) built on **NumPy**:

   * Shape `(4, 13, n)` with `dtype=bool` by default
   * Plane metadata (`plane_names`) so games can interpret each feature plane
   * Helpers to build tensors from sets of cards and to roundtrip back to card sets
3. **End-to-end tests + property tests** validating:

   * All mappings (`Card` ↔ index ↔ tensor positions) are bijective
   * Simple “multi-zone” layouts roundtrip correctly (e.g. hand / stock / discard)
4. **Docs updated**:

   * A focused `docs/core_tensor.md` explaining the encoding
   * `docs/qa.md` extended with the new invariants and coverage
   * `README.md` snippet showing minimal usage

We keep **100% coverage, strict mypy, ruff, pydocstyle, and CI** all green as enforced in M00.

The design aligns with how card RL frameworks encode state: RLCard uses a 52-length vector with each bit tied to a specific card index, grouped by suit and rank, and this pattern generalizes cleanly to a 4×13 grid. ([RLCard][1]) Hearts and Dou Dizhu RL work also rely on 52-length one-hot or 4×13 grid encodings for card combinations, reinforcing this structure as a good “universal” substrate. ([CS230 Deep Learning][2])

---

## Design Decisions (Lock-In for M01)

Cursor should treat these as *constitutional* for UNGAR:

1. **Canonical deck:**

   * 4 suits × 13 ranks = 52 cards (no jokers). ([Stack Overflow][3])
   * Suit order (axis 0): `SPADES`, `HEARTS`, `DIAMONDS`, `CLUBS` (matches common RL encodings where indices 0–12 = spades A–K, 13–25 = hearts, etc.). ([RLCard][1])
   * Rank order (axis 1): `ACE, TWO, THREE, …, TEN, JACK, QUEEN, KING` — purely canonical; *game-specific* rank ordering (e.g. 3..2 in Dou Dizhu, 2..A in some Hearts variants) will be handled in higher layers. ([arXiv][4])

2. **Tensor representation:**

   * Core type: **`numpy.ndarray`** with shape `(4, 13, n)` and `dtype=bool` for presence/absence planes.
   * Wrapped in an immutable `CardTensor` dataclass with:

     * `data: NDArray[np.bool_]`
     * `plane_names: tuple[str, ...]`
   * `data.flags.writeable` should be set to `False` in `__post_init__` so tensors are logically immutable.

   NumPy boolean arrays are efficient for sparse “is this card present?” features, and they compose nicely with higher-level RL frameworks. ([GeeksforGeeks][5])

3. **Plane semantics:**

   * M01 does **not** lock in a global standard set of planes.
   * Instead, we define a lightweight `CardTensorSpec` that ties **plane names** (strings) to axis-2 indices.
   * Higher layers (later milestones) will define standard specs for particular games or feature sets (e.g. `my_hand`, `opponent_hand`, `discard`, `played_this_trick`), similar to how RL Hearts uses multiple 52-bit features for played cards, cards in current trick, etc. ([CS230 Deep Learning][2])

4. **Runtime dependency policy:**

   * M01 introduces **NumPy** as a **runtime dependency** in `pyproject.toml` under `[project.dependencies]` following PEP 621. ([Python Enhancement Proposals (PEPs)][6])
   * Version range: `numpy >=1.26,<3.0` (adjust as Cursor deems safe based on current stable).
   * `requirements-dev.in` should also list `numpy` so `pip-compile` pulls it into `requirements-dev.txt` for local dev. ([NumPy][7])

5. **Testing strategy:**

   * Continue using classic unit tests **plus** property-based tests via Hypothesis for card/tensor bijections. Property-based testing is ideal for guaranteeing invariants across the whole 52-card space. ([hypothesis.readthedocs.io][8])

---

## Phase M01-A – Card Domain Primitives

### Objective

Introduce a **type-safe, immutable representation of cards** and canonical indices, independent of any game rules.

### Files & Modules

* `src/ungar/enums.py`
* `src/ungar/cards.py`
* `tests/test_cards_unit.py`
* `tests/test_cards_properties.py`

### Tasks for Cursor

1. **Create `Suit` and `Rank` enums**

   `src/ungar/enums.py`:

   * `Enum` classes with stable ordering and `__str__` helpers, similar to best practices for card/suit enums in Python. ([Stack Overflow][9])

   ```python
   from __future__ import annotations

   from enum import Enum, auto

   class Suit(Enum):
       SPADES = auto()
       HEARTS = auto()
       DIAMONDS = auto()
       CLUBS = auto()

       def __str__(self) -> str:
           return self.name

   class Rank(Enum):
       ACE = auto()
       TWO = auto()
       THREE = auto()
       FOUR = auto()
       FIVE = auto()
       SIX = auto()
       SEVEN = auto()
       EIGHT = auto()
       NINE = auto()
       TEN = auto()
       JACK = auto()
       QUEEN = auto()
       KING = auto()

       def __str__(self) -> str:
           return self.name
   ```

   * Expose `SUIT_COUNT = 4`, `RANK_COUNT = 13`, `CARD_COUNT = 52` constants.

2. **Create `Card` dataclass and index mapping**

   `src/ungar/cards.py`:

   * Immutable dataclass:

     ```python
     from __future__ import annotations

     from dataclasses import dataclass
     from typing import ClassVar

     from .enums import Rank, Suit

     @dataclass(frozen=True, slots=True)
     class Card:
         suit: Suit
         rank: Rank

         CARD_COUNT: ClassVar[int] = 52

         def to_index(self) -> int:
             """Return canonical 0-based index in [0, 51] for this card."""
             suit_index = list(Suit).index(self.suit)
             rank_index = list(Rank).index(self.rank)
             return suit_index * len(Rank) + rank_index

         @staticmethod
         def from_index(index: int) -> "Card":
             """Create a Card from canonical 0-based index in [0, 51]."""
             if not 0 <= index < Card.CARD_COUNT:
                 msg = f"Card index must be in [0, {Card.CARD_COUNT - 1}], got {index}"
                 raise ValueError(msg)
             suit_index, rank_index = divmod(index, len(Rank))
             suit = list(Suit)[suit_index]
             rank = list(Rank)[rank_index]
             return Card(suit=suit, rank=rank)
     ```

   * Utility functions:

     * `all_cards() -> tuple[Card, ...]` enumerating the deck in index order.
     * `iter_cards()` generator, if helpful.

3. **Unit tests**

   `tests/test_cards_unit.py`:

   * Verify:

     * `SUIT_COUNT == 4`, `RANK_COUNT == 13`, `CARD_COUNT == 52`.
     * `len(all_cards()) == 52` and all elements are unique.
     * `Card.from_index(i).to_index() == i` for all `i in range(52)`.
     * Boundary conditions for `from_index` with negative or ≥52 indices.

4. **Property tests**

   `tests/test_cards_properties.py`:

   * Use Hypothesis to generate random indices and cards, then assert:

     * `from_index(to_index(card)) == card`
     * `to_index(from_index(i)) == i`
   * This matches general best practices for property-based testing. ([hypothesis.readthedocs.io][8])

### Acceptance Criteria

* All new code is fully typed, doc-stringed, and passes `ruff`, `mypy`, `pydocstyle`, and `pytest` with 100% coverage preserved.
* `Card` ↔ index ↔ `(suit, rank)` mapping is **bijective**, validated by unit + property tests.

---

## Phase M01-B – `CardTensor` & Layout Spec

### Objective

Create the **canonical 4×13×n tensor container** and its layout metadata, plus helper APIs to build and inspect tensors.

### Files & Modules

* `src/ungar/tensor.py`
* `tests/test_tensor_shape_and_types.py`
* `tests/test_tensor_roundtrip.py`
* `pyproject.toml` (updated)
* `requirements-dev.in` / `requirements-dev.txt` (updated)

### Tasks for Cursor

1. **Add NumPy as a dependency**

   * In `pyproject.toml`, under `[project]`, add:

     ```toml
     dependencies = [
       "numpy>=1.26,<3.0",
     ]
     ```

     (Adjust the upper bound if needed.)

   * This follows PEP 621’s guidance to list runtime dependencies in `project.dependencies`, ensuring users who install UNGAR also get NumPy. ([Python Enhancement Proposals (PEPs)][6])

   * Add `numpy` to `requirements-dev.in` so `pip-compile` includes it in `requirements-dev.txt`. Best practices recommend keeping dev environments reproducible and using tools like `pip-compile` for pinned dependency files. ([GeeksforGeeks][10])

2. **Define `CardTensorSpec`**

   In `src/ungar/tensor.py`:

   ```python
   from __future__ import annotations

   from dataclasses import dataclass
   from typing import Iterable, Mapping, Sequence, Tuple

   import numpy as np
   from numpy.typing import NDArray

   from .cards import Card
   from .enums import SUIT_COUNT, RANK_COUNT

   BoolTensor = NDArray[np.bool_]

   @dataclass(frozen=True, slots=True)
   class CardTensorSpec:
       """Metadata describing the layout of a CardTensor.

       Attributes:
           plane_names: Names for each feature plane along axis 2,
               e.g. ("my_hand", "opponent_hand", "discard").
       """

       plane_names: Tuple[str, ...]

       @property
       def num_planes(self) -> int:
           return len(self.plane_names)
   ```

3. **Define `CardTensor`**

   ```python
   @dataclass(frozen=True, slots=True)
   class CardTensor:
       """Immutable 4×13×n tensor over the 52-card deck.

       Axis 0: suits, length 4.
       Axis 1: ranks, length 13.
       Axis 2: feature planes, length n.
       """

       data: BoolTensor
       spec: CardTensorSpec

       def __post_init__(self) -> None:
           if self.data.shape != (SUIT_COUNT, RANK_COUNT, self.spec.num_planes):
               msg = (
                   "CardTensor data shape must be (4, 13, n); "
                   f"got {self.data.shape} for {self.spec.num_planes} planes"
               )
               raise ValueError(msg)
           if self.data.dtype != np.bool_:
               msg = f"CardTensor dtype must be bool, got {self.data.dtype}"
               raise TypeError(msg)
           # Enforce logical immutability
           self.data.setflags(write=False)
   ```

   * Classmethods/helpers:

     ```python
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
         """Return the 4×13 slice for a given plane name."""
         plane_index = self.spec.plane_names.index(plane_name)
         return self.data[:, :, plane_index]
     ```

   * Optional bridging flatteners (useful for future RediAI / RL frameworks):

     ```python
     def flat_plane(self, plane_name: str) -> BoolTensor:
         """Return a flattened 52-element vector for a plane (row-major)."""
         return self.plane(plane_name).reshape(-1)

     @classmethod
     def from_flat_plane(
         cls,
         plane_name: str,
         flat: BoolTensor,
     ) -> "CardTensor":
         """Create a single-plane tensor from a 52-element flat boolean vector."""
         if flat.size != SUIT_COUNT * RANK_COUNT:
             msg = f"Flat plane must have size 52, got {flat.size}"
             raise ValueError(msg)
         spec = CardTensorSpec((plane_name,))
         data = flat.reshape(SUIT_COUNT, RANK_COUNT, 1)
         return cls(data=data, spec=spec)
     ```

   This “flat plane” representation aligns with RLCard’s approach, where each card is a specific index in a 52-element vector. ([RLCard][1])

4. **Shape and type tests**

   `tests/test_tensor_shape_and_types.py`:

   * Verify:

     * `CardTensor.empty(["a", "b"]).data.shape == (4, 13, 2)`.
     * `CardTensor.empty([...]).data.dtype == bool`.
     * `CardTensor.empty([...]).data.flags.writeable is False`.
     * Incorrect shapes or dtypes raise `ValueError` / `TypeError` as expected.

5. **Roundtrip tests**

   `tests/test_tensor_roundtrip.py`:

   * Simple hand-style spec:

     ```python
     def test_roundtrip_two_plane_layout() -> None:
         # Example: my_hand vs. discard
         from ungar.cards import Card, all_cards
         from ungar.enums import Suit, Rank
         from ungar.tensor import CardTensor

         my_hand = {
             Card(Suit.SPADES, Rank.ACE),
             Card(Suit.HEARTS, Rank.KING),
         }
         discard = set(all_cards()) - my_hand

         tensor = CardTensor.from_plane_card_map(
             {"my_hand": my_hand, "discard": discard}
         )

         assert set(tensor.cards_in_plane("my_hand")) == my_hand
         assert set(tensor.cards_in_plane("discard")) == discard
     ```

   * Property test: random partitioning of the deck into a few labeled zones, then assert each card appears in *exactly one* of those planes and roundtrips correctly.

### Acceptance Criteria

* All tests under `tests/test_tensor_*` pass.
* Coverage remains at **100%** and all quality gates from M00 stay green.
* A basic end-to-end “build tensor → inspect plane → extract cards” scenario exists and is tested.

---

## Phase M01-C – Invariants & Validation Helpers

### Objective

Add **guardrail helpers** to check tensor invariants, so later game logic can assert strong properties (e.g. “each card is in exactly one zone”).

### Files & Modules

* `src/ungar/tensor.py` (extended)
* `tests/test_tensor_invariants.py`

### Tasks for Cursor

1. **Add validation helpers**

   In `CardTensor`:

   ```python
   def validate_exclusive_planes(self, plane_names: Sequence[str]) -> None:
       """Ensure each card is present in at most one of the given planes.

       Raises:
           ValueError: If any card is present in more than one of the planes.
       """
       indices = [self.spec.plane_names.index(name) for name in plane_names]
       subset = self.data[:, :, indices]  # shape (4, 13, k)
       # Sum across the plane axis; any > 1 indicates a violation.
       counts = subset.sum(axis=2)  # still bool → auto casts to int
       if np.any(counts > 1):
           raise ValueError("Card appears in more than one of the exclusive planes.")
   ```

   Optionally, also:

   ```python
   def validate_partition(self, plane_names: Sequence[str]) -> None:
       """Ensure planes form a complete partition of the deck (each card in exactly one)."""
       indices = [self.spec.plane_names.index(name) for name in plane_names]
       subset = self.data[:, :, indices]
       counts = subset.sum(axis=2)
       if not np.all((counts == 0) | (counts == 1)):
           raise ValueError("At least one card appears in multiple planes.")
       if not np.all(counts == 1):
           raise ValueError("At least one card is not present in exactly one plane.")
   ```

2. **Invariant tests**

   `tests/test_tensor_invariants.py`:

   * Build a simple four-zone layout (`p1_hand`, `p2_hand`, `stock`, `discard`) and:

     * Assert `validate_exclusive_planes` and `validate_partition` pass when partition is correct.
     * Assert they fail when you deliberately double-assign or omit a card.

These helpers become **QA guardrails** for future game-specific layers and help catch mistakes early in workflows.

### Acceptance Criteria

* Invariant helpers exist and are covered by unit tests.
* They do **not** introduce any new dependencies or slow tests (no big random loops).

---

## Phase M01-D – Documentation & QA Updates

### Objective

Document the card domain and tensor encoding so future milestones and external users have a clear, stable target.

### Files & Modules

* `docs/core_tensor.md` (new)
* `docs/index.md` (updated)
* `docs/qa.md` (updated)
* `README.md` (updated)

### Tasks for Cursor

1. **Create `docs/core_tensor.md`**

   Include:

   * Explanation of:

     * 4 suits × 13 ranks = 52 cards.
     * Suit and rank ordering.
     * 4×13×n layout and meaning of each axis.

   * Example diagrams or ASCII layout of the 4×13 grid.

   * Short code examples:

     ```python
     from ungar.cards import Card
     from ungar.enums import Suit, Rank
     from ungar.tensor import CardTensor

     as_card = Card(Suit.SPADES, Rank.ACE)
     tensor = CardTensor.from_plane_card_map({"my_hand": [as_card]})

     print(tensor.plane("my_hand"))       # 4×13 matrix
     print(tensor.flat_plane("my_hand"))  # 52-length vector
     ```

   * Brief note connecting this representation to common RL encodings (e.g. RLCard’s 52-element state vector; Hearts and Dou Dizhu encodings using 52-bit layers or 4×13 grids). ([RLCard][1])

2. **Update `docs/index.md`**

   * Add a “Core Concepts” section linking to:

     * `VISION.md`
     * `docs/core_tensor.md`

3. **Update `docs/qa.md`**

   * Add a subsection “M01 – Card Tensor” listing:

     * New invariants:

       * 52-card bijection (`Card` ↔ index ↔ tensor)
       * Partition validation helpers
     * Testing practices:

       * Hypothesis property tests for card index mapping and tensor roundtrips.
   * Note that coverage is still 100% and CI gates remain identical to M00.

4. **Update `README.md`**

   * Add a “Core Card Tensor” quickstart:

     ````md
     ## Core Card Tensor

     UNGAR represents the 52-card deck as a 4×13×n NumPy tensor:

     * Axis 0: suits (Spades, Hearts, Diamonds, Clubs)
     * Axis 1: ranks (Ace through King)
     * Axis 2: feature planes (game-defined)

     ```python
     from ungar.cards import Card
     from ungar.enums import Suit, Rank
     from ungar.tensor import CardTensor

     my_hand = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
     tensor = CardTensor.from_plane_card_map({"my_hand": my_hand})

     assert set(tensor.cards_in_plane("my_hand")) == set(my_hand)
     ````

     ```
     ```

### Acceptance Criteria

* Docs build cleanly as Markdown and render sensibly in GitHub.
* Any new examples are covered by corresponding tests where applicable (so examples don’t drift).

---

## Handoff Summary for Cursor

When you hand this to Cursor, frame **M01** as:

> **Milestone M01 – Core Card Physics**
>
> Implement phases **M01-A through M01-D**:
>
> 1. **M01-A** – Card Domain Primitives
>
>    * Add `Suit`, `Rank`, and immutable `Card` with canonical 0–51 indexing, plus `all_cards()`.
>    * Add unit + Hypothesis property tests verifying bijections.
> 2. **M01-B** – CardTensor & Layout Spec
>
>    * Add NumPy as a runtime dependency (and to dev requirements).
>    * Implement `CardTensorSpec` + immutable `CardTensor` (4×13×n bool).
>    * Implement builders from plane→cards and helpers `cards_in_plane`, `plane`, `flat_plane`, `from_flat_plane`.
>    * Add end-to-end tests and property tests for tensor roundtrips.
> 3. **M01-C** – Invariants & Guardrails
>
>    * Implement `validate_exclusive_planes` and `validate_partition`.
>    * Add tests that verify correct layouts pass and incorrect ones fail.
> 4. **M01-D** – Documentation & QA
>
>    * Add `docs/core_tensor.md`.
>    * Update `docs/index.md`, `docs/qa.md`, and `README.md` with tensor explanations and examples.
>
> Throughout M01:
>
> * Keep **all M00 quality gates** (ruff, mypy, pydocstyle, pytest+coverage, CI matrix) fully green.
> * Maintain **100% coverage** and strict typing.
> * Ensure `make ci` remains the single command for full local CI.

Once M01 is complete, UNGAR will have a **stable, well-documented tensor substrate** to plug any card game into, and M02 can start defining specific game schemas and simple environments on top of it.

[1]: https://rlcard.org/games.html?utm_source=chatgpt.com "Games in RLCard"
[2]: https://cs230.stanford.edu/projects_spring_2021/reports/9.pdf?utm_source=chatgpt.com "Reinforcement Learning on Hearts - CS230"
[3]: https://stackoverflow.com/questions/31852353/more-efficient-way-to-store-playing-cards-in-bits?utm_source=chatgpt.com "More efficient way to store Playing Cards in bits?"
[4]: https://arxiv.org/pdf/2407.10279?utm_source=chatgpt.com "high-performance end-to-end doudizhu ai integrating bidding"
[5]: https://www.geeksforgeeks.org/python/python-boolean-array-in-numpy/?utm_source=chatgpt.com "Boolean Array in NumPy - Python"
[6]: https://peps.python.org/pep-0621/?utm_source=chatgpt.com "PEP 621 – Storing project metadata in pyproject.toml"
[7]: https://numpy.org/devdocs/dev/depending_on_numpy.html?utm_source=chatgpt.com "For downstream package authors"
[8]: https://hypothesis.readthedocs.io/?utm_source=chatgpt.com "Hypothesis 6.148.2 documentation"
[9]: https://stackoverflow.com/questions/41970795/what-is-the-best-way-to-create-a-deck-of-cards?utm_source=chatgpt.com "python - What is the best way to create a deck of cards?"
[10]: https://www.geeksforgeeks.org/python/best-practices-for-managing-python-dependencies/?utm_source=chatgpt.com "Best Practices for Managing Python Dependencies"
