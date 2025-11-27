from ungar_bridge.training import train_high_card_duel


def test_training_determinism() -> None:
    """Test that training is deterministic with a fixed seed."""
    res1 = train_high_card_duel(num_episodes=50, seed=123)
    res2 = train_high_card_duel(num_episodes=50, seed=123)

    assert res1.rewards == res2.rewards
    assert res1.episode_lengths == res2.episode_lengths


def test_training_shapes() -> None:
    """Test output shapes match configuration."""
    n = 20
    res = train_high_card_duel(num_episodes=n, seed=42)

    assert len(res.rewards) == n
    assert len(res.episode_lengths) == n
    assert res.config["num_episodes"] == n
    assert res.config["seed"] == 42


def test_training_health() -> None:
    """Soft check that training runs without error and produces valid values."""
    res = train_high_card_duel(num_episodes=10, seed=999)

    # Rewards for HighCardDuel are -1.0, 0.0, or 1.0 (float)
    for r in res.rewards:
        assert r in (-1.0, 0.0, 1.0)

    # Episode length for HighCardDuel is effectively 2 steps (P0, P1),
    # but the loop runs from the perspective of one agent?
    # Our simple loop calls `step` until done.
    # HighCardDuel requires 2 moves.
    # Our loop takes legal moves.
    # Step 1: P0 moves.
    # Step 2: P1 moves.
    # So length should be 2.
    for length in res.episode_lengths:
        assert length == 2
