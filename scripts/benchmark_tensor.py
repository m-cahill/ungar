import time
from random import Random

from ungar.cards import all_cards
from ungar.enums import CARD_COUNT
from ungar.tensor import CardTensor


def main(iterations: int = 10_000, seed: int | None = 123) -> None:
    rng = Random(seed)
    cards = list(all_cards())

    start = time.perf_counter()
    for _ in range(iterations):
        rng.shuffle(cards)
        p0 = set(cards[: CARD_COUNT // 2])
        p1 = set(cards[CARD_COUNT // 2 :])
        tensor = CardTensor.from_plane_card_map({"p0": p0, "p1": p1})
        tensor.validate_partition(["p0", "p1"])
    elapsed = time.perf_counter() - start
    print(f"Iterations: {iterations}")
    print(f"Total time: {elapsed:.3f}s")
    if iterations > 0:
        print(f"Per iteration: {elapsed / iterations * 1e6:.2f}µs")


if __name__ == "__main__":
    main()

