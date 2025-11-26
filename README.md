# UNGAR: Universal Neural Grid for Analysis and Research

> **“All card games, one brain.”**

UNGAR is a unified research environment for imperfect-information card games, designed to build a general-purpose "card intelligence" substrate. Instead of treating Poker, Spades, and Hearts as separate environments, UNGAR models them as different rule sets over a single **4×14×n tensor representation** of card physics.

See [VISION.md](./VISION.md) for the high-level goals, architectural philosophy, and the "moonshot" vision of a General Card Player.

## Architecture Note

**UNGAR Core** is designed to be framework-agnostic. It does not depend on RediAI or any specific RL framework. A future **bridge package** (in a separate repository) will provide the integration layer to connect UNGAR environments with the RediAI workflow registry, tournaments, and XAI tools.

## Quickstart

(Assuming Python 3.10+)

```bash
# Clone the repository
git clone https://github.com/m-cahill/ungar.git
cd ungar

# Install development dependencies
python -m pip install -r requirements-dev.txt

# Run the full CI suite (lint, types, tests) locally
make ci
```

### Run a Game
Run a sample High Card Duel episode in Python:

```bash
python -c "from ungar.games.high_card_duel import make_high_card_duel_spec; from ungar.runner import play_random_episode; print(play_random_episode(make_high_card_duel_spec(), seed=42))"
```

## Core Card Tensor

UNGAR represents the 52-card deck (plus optional jokers) as a 4×14×n NumPy tensor:

* Axis 0: suits (Spades, Hearts, Diamonds, Clubs)
* Axis 1: ranks (Ace through King, plus Joker)
* Axis 2: feature planes (game-defined)

```python
from ungar.cards import Card
from ungar.enums import Suit, Rank
from ungar.tensor import CardTensor

my_hand = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
tensor = CardTensor.from_plane_card_map({"my_hand": my_hand})

assert set(tensor.cards_in_plane("my_hand")) == set(my_hand)
```

## Local CI

To run the same checks that GitHub Actions runs:

```bash
python -m pip install -r requirements-dev.txt
make ci
```

## Roadmap

* **M00** – Repo & CI Bootstrap (Done).
* **M01** – UNGAR core card-physics abstraction + 4×14×n tensor encoder (Done).
* **M02** – Game definitions (High Card Duel) & runner interface (Done).
* **M03** – Security & supply-chain hardening: Bandit, pip-audit, cyclonedx SBOM, basic SLSA provenance for artifacts (Done).
* **M04** – RediAI bridge package (separate repo) and Workflow Registry integration.
