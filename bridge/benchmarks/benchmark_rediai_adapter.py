import time
from statistics import mean

from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.rediai_adapter import RediAIUngarAdapter

N_WARMUP = 100
N_RUNS = 1000


def run_benchmark() -> dict:
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    adapter = RediAIUngarAdapter(env)  # uses protocol stub if no RediAI

    # warmup
    env.reset()
    for _ in range(N_WARMUP):
        adapter.encode_state()

    # measure encode_state
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        adapter.encode_state()
        times.append(time.perf_counter() - start)

    avg_ms = mean(times) * 1000
    return {"avg_encode_ms": avg_ms, "runs": N_RUNS}


if __name__ == "__main__":
    result = run_benchmark()
    print(f"Average encode_state time over {result['runs']} runs: {result['avg_encode_ms']:.3f} ms")
