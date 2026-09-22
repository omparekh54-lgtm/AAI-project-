from .environment.simulator import CabDispatchEnv
from .baselines.nearest_cab import nearest_cab_policy

if __name__ == "__main__":
    env = CabDispatchEnv(seed=42, fleet_size=8, day_minutes=180)
    result = env.run(nearest_cab_policy)
    print("Nearest-cab smoke test")
    for key, value in result.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
