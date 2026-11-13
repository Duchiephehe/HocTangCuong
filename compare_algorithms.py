"""Run repeatable evaluations of vacuum-cleaner policies."""

import argparse
import csv
from collections import deque
from pathlib import Path
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

from q_learning import save_training_plot, train_q_learning
from vacuum_env import VacuumGridEnv

Policy = Callable[[VacuumGridEnv], int]
Position = Tuple[int, int]
EpisodeResult = Dict[str, object]

DEFAULT_OBSTACLES = [(1, 1), (2, 2), (3, 1), (3, 3)]


def nearest_dust_policy(env: VacuumGridEnv) -> int:
    """Take a shortest-path step toward the closest remaining dust cell."""
    start = env.robot_pos
    previous: Dict[Position, Tuple[Position, int]] = {}
    visited = {start}
    queue = deque([start])
    target = None
    moves = (
        (env.ACTION_UP, 0, 1),
        (env.ACTION_DOWN, 0, -1),
        (env.ACTION_LEFT, -1, 0),
        (env.ACTION_RIGHT, 1, 0),
    )

    while queue:
        position = queue.popleft()
        if position in env.active_dust:
            target = position
            break
        for action, dx, dy in moves:
            neighbor = (position[0] + dx, position[1] + dy)
            if (
                0 <= neighbor[0] < env.width
                and 0 <= neighbor[1] < env.height
                and neighbor not in env.obstacles
                and neighbor not in visited
            ):
                visited.add(neighbor)
                previous[neighbor] = (position, action)
                queue.append(neighbor)

    if target is None:
        raise RuntimeError("Không tìm thấy đường đi tới vị trí bụi còn lại.")

    while previous[target][0] != start:
        target = previous[target][0]
    return previous[target][1]


POLICIES: Dict[str, Policy] = {
    "Bụi gần nhất (BFS)": nearest_dust_policy,
}


def make_scenarios(episodes: int, seed: int) -> List[List[Position]]:
    """Create one fixed set of dust layouts shared by every policy."""
    rng = random.Random(seed)
    start = (0, 0)
    available = [
        (x, y)
        for y in range(5)
        for x in range(5)
        if (x, y) != start and (x, y) not in DEFAULT_OBSTACLES
    ]
    return [rng.sample(available, 3) for _ in range(episodes)]


def evaluate_algorithms(
    episodes: int = 30,
    seed: int = 42,
    max_steps: int = 100,
    training_episodes: int = 15000,
) -> Tuple[List[EpisodeResult], List[float]]:
    if episodes <= 0:
        raise ValueError("episodes phải là số nguyên dương.")
    if max_steps <= 0:
        raise ValueError("max_steps phải là số nguyên dương.")
    if training_episodes <= 0:
        raise ValueError("training_episodes phải là số nguyên dương.")

    q_table, reward_history = train_q_learning(
        VacuumGridEnv(max_steps=max_steps),
        episodes=training_episodes,
        seed=seed,
    )
    policy_rng = random.Random(seed)

    def q_learning_policy(env: VacuumGridEnv) -> int:
        state = env.get_state()
        values = q_table.get(state, [0.0] * len(VacuumGridEnv.ACTION_NAMES))
        best_value = max(values)
        best_actions = [
            action for action, value in enumerate(values) if value == best_value
        ]
        return policy_rng.choice(best_actions)

    scenarios = make_scenarios(episodes, seed)
    results: List[EpisodeResult] = []
    policies = {
        **POLICIES,
        "Q-Learning": q_learning_policy,
    }
    for algorithm, policy in policies.items():
        for episode, dust_positions in enumerate(scenarios, start=1):
            env = VacuumGridEnv(
                dust_positions=dust_positions,
                max_steps=max_steps,
            )
            total_reward = 0.0
            collisions = 0
            started_at = time.perf_counter()
            done = False
            success = False

            while not done:
                _, reward, terminated, truncated, info = env.step(policy(env))
                total_reward += reward
                collisions += int(info["collision"])
                success = terminated
                done = terminated or truncated

            results.append(
                {
                    "algorithm": algorithm,
                    "episode": episode,
                    "dust_positions": repr(dust_positions),
                    "success": success,
                    "steps": env.current_step,
                    "total_reward": total_reward,
                    "collisions": collisions,
                    "dust_remaining": len(env.active_dust),
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
            )
    return results, reward_history


def summarize(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    algorithms = list(dict.fromkeys(str(row["algorithm"]) for row in results))
    for algorithm in algorithms:
        rows = [row for row in results if row["algorithm"] == algorithm]
        count = len(rows)
        summaries[algorithm] = {
            "success_rate_percent": 100
            * sum(bool(row["success"]) for row in rows)
            / count,
            "average_steps": sum(int(row["steps"]) for row in rows) / count,
            "average_reward": sum(float(row["total_reward"]) for row in rows)
            / count,
            "average_collisions": sum(int(row["collisions"]) for row in rows)
            / count,
            "average_dust_remaining": sum(
                int(row["dust_remaining"]) for row in rows
            )
            / count,
            "average_elapsed_seconds": sum(
                float(row["elapsed_seconds"]) for row in rows
            )
            / count,
        }
    return summaries


def save_results(results: List[EpisodeResult], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "episode",
        "dust_positions",
        "success",
        "steps",
        "total_reward",
        "collisions",
        "dust_remaining",
        "elapsed_seconds",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_comparison_plot(results: List[EpisodeResult], output: Path) -> None:
    import matplotlib.pyplot as plt

    algorithms = list(POLICIES) + ["Q-Learning"]
    summaries = summarize(results)
    metrics = (
        ("success_rate_percent", "Tỷ lệ dọn sạch thành công (%)"),
        ("average_steps", "Số bước trung bình"),
        ("average_reward", "Tổng reward trung bình"),
    )
    figure, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for axis, (metric, title) in zip(axes, metrics):
        values = [summaries[name][metric] for name in algorithms]
        bars = axis.bar(algorithms, values, color=["#F4B400", "#2878B5"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.3)
        axis.tick_params(axis="x", rotation=12)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.1f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    figure.suptitle("So sánh Q-Learning và chính sách bụi gần nhất (BFS)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="So sánh các chính sách trong môi trường robot hút bụi."
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--training-episodes", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("algorithm_comparison.csv"))
    args = parser.parse_args()

    results, reward_history = evaluate_algorithms(
        args.episodes,
        args.seed,
        args.max_steps,
        args.training_episodes,
    )
    save_results(results, args.output)
    training_plot = args.output.with_name(
        f"{args.output.stem}_q_learning_training.png"
    )
    comparison_plot = args.output.with_suffix(".png")
    save_training_plot(reward_history, training_plot)
    save_comparison_plot(results, comparison_plot)

    print(f"So sánh trên {args.episodes} bản đồ (seed={args.seed}):")
    print(
        f"{'Thuật toán':<24} {'Thành công':>12} {'Bước TB':>10} "
        f"{'Reward TB':>12} {'Va chạm TB':>12} {'Bụi còn TB':>12}"
    )
    for algorithm, metrics in summarize(results).items():
        print(
            f"{algorithm:<24} "
            f"{metrics['success_rate_percent']:>11.1f}% "
            f"{metrics['average_steps']:>10.2f} "
            f"{metrics['average_reward']:>12.2f} "
            f"{metrics['average_collisions']:>12.2f} "
            f"{metrics['average_dust_remaining']:>12.2f}"
        )
    print(f"\nKết quả từng lượt đã lưu tại: {args.output}")
    print(f"Biểu đồ so sánh đã lưu tại: {comparison_plot}")
    print(f"Biểu đồ huấn luyện Q-Learning đã lưu tại: {training_plot}")

    import matplotlib.pyplot as plt

    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
