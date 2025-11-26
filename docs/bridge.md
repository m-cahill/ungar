# UNGAR Bridge & Adapters

The `ungar-bridge` package provides a layer of **adapters** that connect the pure-logic `ungar` core to external AI, RL, or UI frameworks.

## Architecture

*   **Core (`ungar`):** Pure Python/NumPy. Logic, rules, tensors. No external dependencies.
*   **Bridge (`ungar-bridge`):** Depends on `ungar` + external frameworks (optional). Contains adapters.

## Adapters

Adapters implement the `BridgeAdapter` interface, transforming UNGAR state/moves into the format required by the consumer.

### RL Adapter (Planned)
Will map `HighCardDuel` to a Gym-like `Env`.

## Creating a New Adapter

Implement the `BridgeAdapter` protocol in `ungar_bridge/adapter_base.py`:

```python
class BridgeAdapter(Protocol):
    def initialize(self, config: dict) -> None: ...
    def state_to_external(self, state: Any) -> Any: ...
    def external_to_move(self, external_action: Any) -> Move: ...
```

