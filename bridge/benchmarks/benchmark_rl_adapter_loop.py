from statistics import mean
from time import perf_counter

from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.rl_adapter import UngarGymEnv

EPISODES = 500

def run_benchmark() -> dict:
    spec = make_high_card_duel_spec()
    game_env = GameEnv(spec)
    env = UngarGymEnv(game_env)
    
    lengths = []
    times = []
    
    for _ in range(EPISODES):
        t0 = perf_counter()
        # Reset returns (obs, info)
        _, _ = env.reset()
        length = 0
        while True:
            actions = env.legal_actions()
            if not actions:
                break
            # Step returns (obs, reward, terminated, truncated, info)
            # We pick the first legal action (greedy/dumb)
            _, _, terminated, truncated, _ = env.step(actions[0])
            length += 1
            if terminated or truncated:
                break
        times.append(perf_counter() - t0)
        lengths.append(length)
        
    return {
        "episodes": EPISODES,
        "avg_episode_ms": mean(times) * 1000,
        "avg_len": mean(lengths),
    }

if __name__ == "__main__":
    result = run_benchmark()
    print(
        f"Episodes: {result['episodes']}, "
        f"avg episode time: {result['avg_episode_ms']:.3f} ms, "
        f"avg length: {result['avg_len']:.2f}"
    )

