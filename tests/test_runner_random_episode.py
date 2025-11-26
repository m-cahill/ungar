"""Tests for the simulation runner."""

from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar.runner import play_random_episode


def test_high_card_duel_random_episode() -> None:
    """Test that runner can play HighCardDuel to completion."""
    spec = make_high_card_duel_spec()
    episode = play_random_episode(spec, seed=123)

    # HighCardDuel always has exactly 2 moves (reveal, reveal)
    assert len(episode.moves) == 2
    assert len(episode.states) == 3  # Start, after P0, after P1

    # Check rewards shape
    assert len(episode.rewards) == 2
    # Sum should be 0.0 (zero sum game)
    assert sum(episode.rewards) == 0.0


def test_runner_determinism() -> None:
    """Test that runner is deterministic with seed."""
    spec = make_high_card_duel_spec()
    ep1 = play_random_episode(spec, seed=42)
    ep2 = play_random_episode(spec, seed=42)
    ep3 = play_random_episode(spec, seed=999)

    assert ep1.moves == ep2.moves
    assert ep1.rewards == ep2.rewards

    # With high probability, different seeds might yield different outcomes
    # (though with 2 players and 1 card each, collision is possible, but full state sequence likely differs)
    # Actually, hands will likely differ.
    assert ep1.states[0] != ep3.states[0]
