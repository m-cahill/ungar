# Universal Neural Grid for Analysis and Research (UNGAR)

Implementation of [VISION.md](VISION.md).

## Project Structure

```text
src/ungar/
├── cards.py      # Core Card/Suit/Rank primitives
├── enums.py      # Constants (SUIT_COUNT=4, RANK_COUNT=14)
├── game.py       # GameState, GameSpec, Move protocols
├── runner.py     # Simulation runner (play_random_episode)
├── tensor.py     # 4x14xN CardTensor implementation
└── games/        # Game implementations
    └── high_card_duel.py

bridge/
├── src/ungar_bridge/
│   ├── adapter_base.py    # BridgeAdapter interface
│   ├── noop_adapter.py    # Dummy implementation
│   ├── rediai_adapter.py  # RediAI EnvAdapter integration
│   └── rl_adapter.py      # Gym-like environment wrapper
└── examples/
    └── demo_rediai.py     # RediAI workflow demo
```

## Data Schema

### Core Tensor
*   **Dimensions:** 4 × 14 × N
    *   **Axis 0 (Suits):** SPADES, HEARTS, DIAMONDS, CLUBS
    *   **Axis 1 (Ranks):** ACE, TWO...KING, JOKER
    *   **Axis 2 (Planes):** Game-specific feature planes (N)
*   **Dtype:** Boolean (True = card present in plane).

### Game Interfaces
*   **GameState:** Protocol defining `current_player`, `legal_moves`, `apply_move`, `is_terminal`, `returns`, `to_tensor`.
*   **Move:** Simple `(id, name)` tuple.

## Implemented Games

### High Card Duel
*   **Players:** 2
*   **Deck:** Standard 52 (Jokers supported but not used in default deal).
*   **Rules:** Each player gets 1 card. Reveal. High card wins.
*   **Tensor Planes:** `my_hand`, `opponent_hand`, `unseen`.

## Milestones

*   **M00:** Foundation (Repo, CI).
*   **M01:** Core Tensor (4x14 grid, partitioning).
*   **M02:** Game Interfaces & High Card Duel.
*   **M03:** Security & Supply Chain (Bandit, SBOM, Release Flow).
*   **M04:** Bridge Package & External Integration.
*   **M05:** RediAI Integration (Bridge-Level).
*   **M06:** Bridge Quality & Coverage Hardening.

## Documentation

*   [bridge/README.md](bridge/README.md): Bridge package usage.
*   [docs/bridge_rediai.md](docs/bridge_rediai.md): Detailed RediAI integration docs.
