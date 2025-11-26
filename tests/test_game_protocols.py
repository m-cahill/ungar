"""Tests for core game protocols and environment."""

import ast
import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import pytest
from ungar.game import (
    GameEnv,
    GameSpec,
    GameState,
    IllegalMoveError,
    Move,
)
from ungar.tensor import CardTensor


@dataclass(frozen=True, slots=True)
class MockState(GameState):
    """A minimal mock state for testing protocols."""

    counter: int = 0
    max_steps: int = 3

    def current_player(self) -> int:
        return 0

    def legal_moves(self) -> Sequence[Move]:
        if self.is_terminal():
            return ()
        return (Move(id=0, name="increment"),)

    def is_terminal(self) -> bool:
        return self.counter >= self.max_steps

    def apply_move(self, move: Move) -> "MockState":
        if self.is_terminal():
            raise IllegalMoveError("Terminal")
        if move.id != 0:
            raise IllegalMoveError(f"Bad move: {move}")
        return MockState(counter=self.counter + 1, max_steps=self.max_steps)

    def returns(self) -> Tuple[float, ...]:
        if not self.is_terminal():
            raise RuntimeError("Not terminal")
        return (1.0,)

    def to_tensor(self, player: int) -> CardTensor:
        return CardTensor.empty(("dummy",))


@dataclass(frozen=True, slots=True)
class MockSpec(GameSpec):
    def initial_state(self, seed: int | None = None) -> GameState:
        return MockState()


def test_game_env_flow() -> None:
    """Test standard environment reset/step cycle."""
    spec = MockSpec(name="mock", num_players=1)
    env = GameEnv(spec)

    state = env.reset()
    assert isinstance(state, MockState)
    assert state.counter == 0
    assert not state.is_terminal()

    # Step 1
    move = Move(id=0, name="increment")
    state, rewards, done, info = env.step(move)
    # Cast to MockState to access specific fields not in Protocol
    assert isinstance(state, MockState)
    assert state.counter == 1
    assert not done
    assert rewards == ()

    # Step 2
    state, rewards, done, info = env.step(move)
    assert isinstance(state, MockState)
    assert state.counter == 2
    assert not done

    # Step 3 (Terminal)
    state, rewards, done, info = env.step(move)
    assert isinstance(state, MockState)
    assert state.counter == 3
    assert done
    assert rewards == (1.0,)


def test_illegal_move() -> None:
    """Test that illegal moves raise appropriate errors."""
    spec = MockSpec(name="mock", num_players=1)
    env = GameEnv(spec)
    env.reset()

    bad_move = Move(id=999, name="bad")
    with pytest.raises(IllegalMoveError):
        env.step(bad_move)


def test_step_without_reset() -> None:
    """Test that stepping before reset raises RuntimeError."""
    spec = MockSpec(name="mock", num_players=1)
    env = GameEnv(spec)
    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(Move(id=0, name="increment"))


def test_no_magic_numbers() -> None:
    """Guardrail: Ensure no hard-coded 4/14/56/52 constants exist in source code.

    This scans the AST of all source files to prevent regression to 4x13
    or other magic numbers. Allowed exceptions:
    - 0, 1 (common logic)
    - Small offsets like 2, 3 in logic (though ideally named)
    - Test files are excluded (they often assert specific values)
    """
    forbidden = {4, 13, 14, 52, 56}
    source_dir = "src/ungar"

    for root, _, files in os.walk(source_dir):
        for filename in files:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(root, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, int):
                    if node.value in forbidden:
                        # Allow defining the constants themselves in enums.py
                        if filename == "enums.py":
                            continue
                        msg = (
                            f"Found forbidden magic number {node.value} in {filename}:{node.lineno}. "
                            "Use constants from ungar.enums (RANK_COUNT, SUIT_COUNT, etc.)."
                        )
                        pytest.fail(msg)
