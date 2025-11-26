Here’s how I’d shape **M02** so it builds cleanly on the 4×14 work, stays small per phase, and keeps everything end-to-end tested.

You’re already fully 4×14-compliant and green across CI with 100% coverage.
The README now explicitly defines UNGAR as a **4×14×n** substrate and sets M02 as “Game definitions for 1–2 simple games (e.g., High Card Duel + simple trick-taker)”. 

M02 will be: **“Game Interfaces & High Card Duel”** + a first **tensor performance baseline**.

---

## M02 – Game Interfaces & High Card Duel

### 🎯 M02 Goals

By the end of M02, the `ungar` repo should have:

1. A **framework-agnostic game interface layer**:

   * `GameState` and `GameSpec` (and optionally `GameEnv`) protocols
   * Consistent handling of `current_player`, legal moves, terminal states, rewards, and tensor observation
2. A **reference toy game**:

   * `HighCardDuel` implemented on top of the 4×14×n tensor
   * Invariants enforced via `CardTensor` guardrails (partition/exclusive planes)
3. A **basic game runner**:

   * A tiny simulation loop that can play random episodes and prove the interfaces actually work
4. A **first performance baseline**:

   * Simple benchmark for `CardTensor` operations (create tensors, validate partitions, etc.)
5. All of this under:

   * **100% coverage**, strict typing, and all M00/M01 CI gates still green
   * No reintroduction of 4×13 anywhere (we’ll add one extra guardrail on that)

The design should echo **RLCard’s “configurable environment” approach** and **OpenSpiel’s general game API**: generic state/env interfaces, game-specific logic plugged in behind them.

---

## Phase M02-A – Core Game Interfaces

**Objective:** Introduce a **minimal but extensible** game abstraction layer that will work for all future card games.

### New module

* `src/ungar/game.py` (name can be `core/game.py` if you prefer a subpackage)

### Types & Protocols

Define a few shared types:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .tensor import CardTensor

PlayerId = int
Reward = float
MoveId = int
```

1. **Move abstraction**

Start simple: for M02, a light dataclass is enough:

```python
@dataclass(frozen=True, slots=True)
class Move:
    """A game move in UNGAR's core model.

    Attributes:
        id: Stable integer identifier (0..N-1 per game).
        name: Human-readable name, e.g. "reveal".
    """

    id: MoveId
    name: str
```

(We can extend later if we need richer payloads.)

2. **GameState Protocol**

Inspired by OpenSpiel’s `State` interface (`CurrentPlayer`, `LegalActions`, `ApplyAction`, `IsTerminal`, `Returns`, `InformationStateTensor`), and similar RL frameworks.

```python
class GameState(Protocol):
    """Generic interface for turn-based, finite card games."""

    def current_player(self) -> PlayerId:
        """Return the ID of the player whose turn it is (0..N-1)."""

    def legal_moves(self) -> Sequence[Move]:
        """Return all legal moves for the current player."""

    def is_terminal(self) -> bool:
        """Return True if the game is over."""

    def apply_move(self, move: Move) -> "GameState":
        """Return the next state after applying the given move.

        Raises:
            IllegalMoveError: If the move is not legal in this state.
        """

    def returns(self) -> Tuple[Reward, ...]:
        """Return final rewards per player.

        Precondition:
            is_terminal() is True.
        """

    def to_tensor(self, player: PlayerId) -> CardTensor:
        """Return the 4×14×n observation tensor for `player`."""
```

3. **IllegalMoveError**

```python
class IllegalMoveError(Exception):
    """Raised when an illegal move is applied to a GameState."""
```

4. **GameSpec & GameEnv**

RLCard and similar toolkits expose environment-level methods like `reset()` / `step(action)` for RL use.

```python
@dataclass(frozen=True, slots=True)
class GameSpec:
    """Static description of a game."""

    name: str
    num_players: int

    def initial_state(self, seed: int | None = None) -> GameState:
        """Return a fresh initial state, optionally seeded."""
```

Optional (but recommended) thin environment wrapper:

```python
@dataclass
class GameEnv:
    """Simple environment wrapper around a GameSpec."""

    spec: GameSpec
    state: GameState | None = None

    def reset(self, seed: int | None = None) -> GameState:
        self.state = self.spec.initial_state(seed=seed)
        return self.state

    def step(self, move: Move) -> tuple[GameState, Tuple[Reward, ...], bool, Mapping[str, Any]]:
        if self.state is None:
            msg = "Environment must be reset() before step()."
            raise RuntimeError(msg)
        next_state = self.state.apply_move(move)
        self.state = next_state
        done = next_state.is_terminal()
        rewards = next_state.returns() if done else tuple()
        info: Mapping[str, Any] = {}
        return next_state, rewards, done, info
```

### Tests & Guardrails

* `tests/test_game_protocols.py`:

  * Use a tiny **mock game** (hardcoded state class) to show:

    * `GameSpec.initial_state()` returns something implementing `GameState`.
    * `GameEnv.reset()` and `step()` behave correctly.
    * `IllegalMoveError` is raised when expected.

* **4×14 guardrail:**
  Add a small regression test that asserts **all** production code uses `SUIT_COUNT`, `RANK_COUNT`, and `CARD_COUNT` rather than hard-coded 4/14/56, except where you explicitly test those constants. This prevents backsliding to 4×13 magic numbers.

**Acceptance for M02-A**

* New interfaces exist and are fully covered by tests.
* CI still green; `make ci` passes.
* No new dependency added.

---

## Phase M02-B – Implement `HighCardDuel` Game

**Objective:** Implement a simple 2-player card game fully on top of the tensor substrate, proving that the game interface and 4×14 deck work together.

### Game Rules (simple version)

**HighCardDuel**:

* 2 players, IDs 0 and 1.
* At start, each player is dealt **1 private card** from the 56-card deck (Joker treated as highest rank for tie-break).
* Game flow:

  1. Start state: both players have their cards, nothing revealed.
  2. Player 0 turn: must play `Move(id=0, name="reveal")`.
  3. Player 1 turn: same.
  4. After both have revealed, the game is **terminal**; higher card wins.
  5. Rewards: `(1.0, -1.0)` if player 0 wins, `(-1.0, 1.0)` if player 1 wins, `(0.0, 0.0)` on exact tie.

This is intentionally trivial on decision-making, but it exercises:

* Dealing from the 4×14 deck
* Private vs public information
* GameState transitions and `IllegalMoveError`
* Tensor representation per player

### Implementation

New module:

* `src/ungar/games/high_card_duel.py`

Key pieces:

1. **Config**

```python
from dataclasses import dataclass
from random import Random
from typing import Tuple

from ungar.cards import Card, all_cards
from ungar.enums import CARD_COUNT, Rank, Suit
from ungar.game import GameSpec, GameState, Move, IllegalMoveError
from ungar.tensor import CardTensor

HIGH_CARD_DUEL_NAME = "high_card_duel"
NUM_PLAYERS = 2
```

2. **Card ranking helper** (game-specific)

```python
def high_card_value(card: Card) -> int:
    """Return a numeric value for ranking cards in HighCardDuel.

    Joker is treated as highest rank.
    """
    # Example: use Rank enum order, but bump Joker to top.
    if card.rank is Rank.JOKER:
        return 100
    return list(Rank).index(card.rank)
```

3. **State dataclass**

```python
@dataclass(frozen=True, slots=True)
class HighCardDuelState(GameState):
    hands: Tuple[Card, Card]
    revealed: Tuple[bool, bool]
    _current_player: int  # 0 or 1, -1 when terminal

    def current_player(self) -> int:
        return self._current_player

    def legal_moves(self) -> Sequence[Move]:
        if self.is_terminal():
            return ()
        return (Move(id=0, name="reveal"),)

    def is_terminal(self) -> bool:
        return all(self.revealed)

    def apply_move(self, move: Move) -> "HighCardDuelState":
        if self.is_terminal():
            raise IllegalMoveError("Game is already terminal.")
        legal = self.legal_moves()
        if move not in legal:
            raise IllegalMoveError(f"Illegal move {move} in state {self!r}")

        # Reveal for current player
        revealed = list(self.revealed)
        revealed[self._current_player] = True
        # Next player or terminal
        next_player = 1 - self._current_player
        if all(revealed):
            next_player = -1

        return HighCardDuelState(
            hands=self.hands,
            revealed=tuple(revealed),
            _current_player=next_player,
        )

    def returns(self) -> Tuple[float, float]:
        if not self.is_terminal():
            raise RuntimeError("Returns only defined for terminal states.")

        v0 = high_card_value(self.hands[0])
        v1 = high_card_value(self.hands[1])
        if v0 > v1:
            return (1.0, -1.0)
        if v1 > v0:
            return (-1.0, 1.0)
        return (0.0, 0.0)

    def to_tensor(self, player: int) -> CardTensor:
        # Use planes like: my_hand, opponent_hand, unseen
        from ungar.tensor import CardTensor

        p0_card, p1_card = self.hands
        all_deck = set(all_cards())
        my = {self.hands[player]}
        opp = {self.hands[1 - player]} if self.revealed[1 - player] else set()
        unseen = all_deck - my - opp

        return CardTensor.from_plane_card_map(
            {
                "my_hand": my,
                "opponent_hand": opp,
                "unseen": unseen,
            }
        )
```

4. **Spec implementation**

```python
@dataclass(frozen=True, slots=True)
class HighCardDuelSpec(GameSpec):
    def initial_state(self, seed: int | None = None) -> GameState:
        rng = Random(seed)
        deck = list(all_cards())
        rng.shuffle(deck)
        hands = (deck[0], deck[1])
        return HighCardDuelState(
            hands=hands,
            revealed=(False, False),
            _current_player=0,
        )


def make_high_card_duel_spec() -> HighCardDuelSpec:
    return HighCardDuelSpec(name=HIGH_CARD_DUEL_NAME, num_players=NUM_PLAYERS)
```

### Tests

* `tests/test_high_card_duel_unit.py`:

  * Deterministic behavior with fixed seed.
  * Rewards for known card pairs (including Joker).
  * `IllegalMoveError` when:

    * Playing a move after terminal
    * Playing a move that isn’t `reveal`

* `tests/test_high_card_duel_tensor.py`:

  * For a given state:

    * `validate_partition(["my_hand", "opponent_hand", "unseen"])` passes.
    * Each card appears in exactly one plane.
    * For unrevealed opponent card, `to_tensor` for the player shows it in `unseen`, not in `opponent_hand`.

**Acceptance for M02-B**

* `HighCardDuelSpec` and `HighCardDuelState` implement `GameSpec` / `GameState`.
* Tests confirm end-to-end: `GameEnv(make_high_card_duel_spec())` can `reset()`, `step(reveal)`, `step(reveal)`, and end in a terminal state with valid rewards.
* All tensor invariants and 4×14 assumptions remain enforced.

---

## Phase M02-C – Simple Game Runner & Random Episodes

**Objective:** Build a tiny **simulation runner** to exercise the interfaces in a more “real” loop and catch workflow errors early.

### New module

* `src/ungar/runner.py`

### Implementation

```python
from dataclasses import dataclass
from random import Random
from typing import List, Tuple

from .game import GameEnv, GameSpec, Move


@dataclass
class Episode:
    states: List[object]
    moves: List[Move]
    rewards: Tuple[float, ...]


def play_random_episode(spec: GameSpec, seed: int | None = None) -> Episode:
    rng = Random(seed)
    env = GameEnv(spec=spec)
    state = env.reset(seed=seed)

    states: List[object] = [state]
    moves: List[Move] = []

    while not state.is_terminal():
        legal = state.legal_moves()
        if not legal:
            raise RuntimeError("Non-terminal state with no legal moves.")
        move = rng.choice(legal)
        state, rewards, done, info = env.step(move)
        states.append(state)
        moves.append(move)
        if done:
            return Episode(states=states, moves=moves, rewards=rewards)

    # If we somehow started in terminal (should not happen), handle here:
    return Episode(states=states, moves=moves, rewards=state.returns())
```

### Tests

* `tests/test_runner_random_episode.py`:

  * For `HighCardDuelSpec`, `play_random_episode` always:

    * Terminates in ≤ 2 steps.
    * Produces a reward tuple of length 2.
    * Uses only legal moves at each step.
  * (Optionally) sample many seeds and assert the empirical average reward is ~0 (game is symmetric), which is a nice sanity check for dealing logic.

This is very similar in spirit to RLCard and OpenSpiel’s basic “loop over episodes and steps” examples used to test environments.

**Acceptance for M02-C**

* Runner exists and is used in tests.
* Any bug in `current_player`, `legal_moves`, or `apply_move` will show up as test failures.

---

## Phase M02-D – Performance Baseline & Documentation

**Objective:** Address the M01 “performance baseline” opportunity and document the new layer.

### 1. Tensor Micro-benchmark

New script:

* `scripts/benchmark_tensor.py`

Example:

```python
import time
from random import Random

from ungar.cards import all_cards
from ungar.tensor import CardTensor
from ungar.enums import CARD_COUNT

def main(iterations: int = 10_000, seed: int | None = 123) -> None:
    rng = Random(seed)
    cards = list(all_cards())

    start = time.perf_counter()
    for _ in range(iterations):
        rng.shuffle(cards)
        p0 = set(cards[:CARD_COUNT // 2])
        p1 = set(cards[CARD_COUNT // 2 :])
        tensor = CardTensor.from_plane_card_map({"p0": p0, "p1": p1})
        tensor.validate_partition(["p0", "p1"])
    elapsed = time.perf_counter() - start
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.3f}s")
    print(f"Per iteration: {elapsed / iterations * 1e6:.2f}µs")

if __name__ == "__main__":
    main()
```

* Not part of CI; just a **manual tool**.
* Document in `docs/qa.md` how to run it and what typical numbers look like.

This responds to the audit’s note that performance is currently “unrated” and should be baselined. 

### 2. Documentation updates

* **`docs/core_tensor.md`**

  * Add a small section “Game Integration” showing how `HighCardDuelState.to_tensor()` maps state zones (`my_hand`, `opponent_hand`, `unseen`) to CardTensor planes.

* **New doc:** `docs/games_high_card_duel.md`

  * Rules summary.
  * State diagram showing:

    * Start → P0 reveal → P1 reveal → terminal.
  * Examples of tensor slices from each player’s perspective.

* **`docs/index.md`**

  * Add “Games” section linking to `games_high_card_duel.md`.

* **`docs/qa.md`**

  * Add “M02 – Game Definitions & Runner” section:

    * GameState/GameSpec invariants.
    * Simulation runner.
    * Tensor performance baseline: how to run `benchmark_tensor.py` and interpret results.

* **`README.md`**

  * Under “Quickstart”, add a snippet:

    ```bash
    # Run a sample HighCardDuel episode in Python
    python -c "from ungar.games.high_card_duel import make_high_card_duel_spec; \
               from ungar.runner import play_random_episode; \
               print(play_random_episode(make_high_card_duel_spec(), seed=42))"
    ```

  * This shows a complete “from install to episode” loop, similar to how RLCard and OpenSpiel demos show simple episodes in documentation.

**Acceptance for M02-D**

* Benchmark script exists and runs without errors.
* Docs updated; no broken links.
* QA doc clearly records:

  * New invariants
  * Runner behavior
  * Where to find performance data

---

## Handoff Summary for Cursor

When you paste this to Cursor, you can frame it as:

> **Milestone M02 – Game Interfaces & High Card Duel**
>
> Implement phases **M02-A → M02-D**:
>
> 1. **M02-A – Game Interfaces**
>
>    * Add `GameState`, `GameSpec`, `GameEnv`, `Move`, `IllegalMoveError`.
>    * Tests: mock game covers the protocol; guardrail to avoid hard-coded 4/14/56 outside constants.
> 2. **M02-B – HighCardDuel Game**
>
>    * Implement `HighCardDuelSpec` / `HighCardDuelState` on top of the 4×14 deck.
>    * Use `CardTensor` for `to_tensor(player)` with partition invariants.
>    * Unit & property tests for dealing, transitions, rewards, and tensor layout.
> 3. **M02-C – Runner & Random Episodes**
>
>    * Implement `play_random_episode(spec, seed)` and test it against HighCardDuel.
>    * Ensure episodes always terminate, only legal moves are used, and rewards have correct shape.
> 4. **M02-D – Performance & Docs**
>
>    * Add `scripts/benchmark_tensor.py` and document how to run it.
>    * Update docs (`core_tensor`, `games_high_card_duel`, `qa`, `README`) to reflect the new layer.
>
> Throughout:
>
> * Keep **CI fully green** (`make ci` = single source of truth).
> * Maintain **100% coverage** and strict mypy/ruff/pydocstyle.
> * Ensure no regressions to 4×13 anywhere in code or docs.

Once M02 is in the books, you’ll have **actual games** running on UNGAR’s 4×14 substrate and a small performance picture—perfect setup for M03’s security & supply-chain hardening and eventually the RediAI bridge.
