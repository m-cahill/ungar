# Training Gin Rummy

UNGAR provides a fully compliant Gin Rummy environment with RediAI integration.

## Game Mechanics

*   **Players:** 2
*   **Deck:** Standard 52-card deck.
*   **Deal:** 10 cards per player.
*   **Goal:** Form melds (sets/runs) to minimize deadwood points.
*   **Knock:** Can end round if deadwood ≤ 10.
*   **Gin:** Knocking with 0 deadwood (+25 bonus).

## Tensor Representation

The observation tensor (4×14×N) includes specialized planes for Gin Rummy:

*   `my_hand`: Cards currently held.
*   `in_discard`: Cards visible in discard pile.
*   `top_discard`: Flag for the top-most discard (drawable).
*   `legal_draw_discard`: Mask indicating if drawing from discard is legal.
*   `legal_discard`: Mask indicating cards legal to discard.
*   `legal_knock`: Mask indicating cards legal to discard *and* knock.

## Training

Use the RediAI bridge to train agents and generate XAI artifacts.

```python
from ungar_bridge.rediai_gin import train_gin_rummy_rediai

# Train for 1000 episodes
# Generates "ungar_gin_overlays.json" and "ungar_reward_decomposition.json"
result = await train_gin_rummy_rediai(num_episodes=1000, record_overlays=True)
```

## Reward Structure

Rewards are decomposed into:
*   `game_score`: Points from Gin/Knock/Undercut logic.
*   `baseline`: 0.0 (placeholder for value baseline).

