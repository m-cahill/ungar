# Core Card Tensor

UNGAR anchors all game states to a single, canonical **4×14×n tensor representation**. This ensures that agents, evaluators, and visualization tools can be reused across different card games without modification.

## The 4×14 Grid

The first two dimensions of the tensor always represent the card universe (standard deck + potential jokers):

*   **Axis 0 (Suits):** Length 4. Order: `SPADES`, `HEARTS`, `DIAMONDS`, `CLUBS`.
*   **Axis 1 (Ranks):** Length 14. Order: `ACE`, `TWO`, `THREE`, ..., `KING`, `JOKER`.

Each cell `(suit, rank)` corresponds to a unique physical card slot. For standard 52-card games, the `JOKER` column is unused (all false).

## The Feature Axis (n)

The third dimension (**Axis 2**) represents **feature planes**. Each plane is a 4×14 boolean grid describing a specific property or location.

For example, a game might define a tensor with 3 planes:
1.  `my_hand`: Cards currently held by the agent.
2.  `public_cards`: Cards face-up on the table.
3.  `discard_pile`: Cards in the discard pile.

A value of `True` at `(s, r, p)` means "Card `(s, r)` is present in plane `p`".

## Usage Example

```python
from ungar.cards import Card
from ungar.enums import Suit, Rank
from ungar.tensor import CardTensor

# Create some cards
ace_spades = Card(Suit.SPADES, Rank.ACE)
king_hearts = Card(Suit.HEARTS, Rank.KING)

# Define a layout with one plane "my_hand"
tensor = CardTensor.from_plane_card_map({
    "my_hand": [ace_spades, king_hearts]
})

# Access the 4x14 boolean grid for "my_hand"
grid = tensor.plane("my_hand")
print(grid.shape)  # (4, 14)

# Check presence
assert grid[0, 0] == True   # Spades (0), Ace (0) is present
assert grid[1, 12] == True  # Hearts (1), King (12) is present
```

## Game Integration

Games implement the `GameState` protocol, which includes a `to_tensor(player)` method. This method maps the game's internal state to a `CardTensor` from the perspective of a specific player.

For example, in `HighCardDuel`:
*   `my_hand`: Contains the card held by the requesting player.
*   `opponent_hand`: Contains the opponent's card **only if it has been revealed**.
*   `unseen`: Contains all other cards (deck + unrevealed opponent card).

This partitioning ensures that the tensor perfectly represents the information set available to the agent, maintaining hidden information properties.

## Relation to RL Frameworks

This structure aligns with state-of-the-art reinforcement learning encodings:
*   **RLCard:** Often uses a flattened 52-bit vector. UNGAR supports this via `flat_plane()` and `from_flat_plane()` (flattening to 56 bits).
*   **DeepStack / Pluribus:** Use similar card buckets or abstractions that map cleanly to this grid.

By standardizing on `(4, 14, n)`, UNGAR enables the use of Convolutional Neural Networks (CNNs) to learn spatial patterns (e.g., flushes, straights) that are invariant across games.
