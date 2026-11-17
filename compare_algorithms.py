"""Run repeatable evaluations of vacuum-cleaner policies."""

import argparse
import csv
from collections import deque
from pathlib import Path
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

from vacuum_env import VacuumGridEnv

Policy = Callable[[VacuumGridEnv], int]
Position = Tuple[int, int]
EpisodeResult = Dict[str, object]

DEFAULT_OBSTACLES = [(1, 1), (2, 2), (3, 1), (3, 3)]


def make_obstacles(grid_size: int = 5) -> List[Position]:
    """Create the default obstacle layout for the requested grid size."""
    obstacles = []
    for x, y in DEFAULT_OBSTACLES:
        if 0 <= x < grid_size and 0 <= y < grid_size:
            obstacles.append((x, y))
    if len(obstacles) != len(DEFAULT_OBSTACLES):
        # keep the classic obstacle pattern even when the grid is slightly smaller
        obstacles = [
            (x, y)
            for x, y in DEFAULT_OBSTACLES
            if 0 <= x < grid_size and 0 <= y < grid_size
        ]
    return obstacles


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


def make_scenarios(
    episodes: int, seed: int, grid_size: int = 5, num_dust: int = 3
) -> List[List[Position]]:
    """Create one fixed set of dust layouts shared by every policy."""
    rng = random.Random(seed)
    start = (0, 0)
    obstacles = set(make_obstacles(grid_size))
    available = [
        (x, y)
        for y in range(grid_size)
        for x in range(grid_size)
        if (x, y) != start and (x, y) not in obstacles
    ]
    if len(available) < num_dust:
        raise ValueError(
            f"Số ô trống khả dụng ({len(available)}) nhỏ hơn num_dust ({num_dust})."
        )
    return [rng.sample(available, num_dust) for _ in range(episodes)]


def evaluate_algorithms(
    episodes: int = 30,
    seed: int = 42,
    max_steps: int = 100,
    grid_size: int = 5,
    num_dust: int = 3,
) -> List[EpisodeResult]:
    if episodes <= 0:
        raise ValueError("episodes phải là số nguyên dương.")
    if max_steps <= 0:
        raise ValueError("max_steps phải là số nguyên dương.")

    scenarios = make_scenarios(episodes, seed, grid_size=grid_size, num_dust=num_dust)
    results: List[EpisodeResult] = []
    for algorithm, policy in POLICIES.items():
        for episode, dust_positions in enumerate(scenarios, start=1):
            env = VacuumGridEnv(
                width=grid_size,
                height=grid_size,
                num_dust=num_dust,
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
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
    summaries: Dict[str, Dict[str, float]] = {}
    for algorithm in POLICIES:
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


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="So sánh các chính sách trong môi trường robot hút bụi."
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--num-dust", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("algorithm_comparison.csv"))
    args = parser.parse_args()

    results = evaluate_algorithms(
        args.episodes,
        args.seed,
        args.max_steps,
        grid_size=args.grid_size,
        num_dust=args.num_dust,
    )
    save_results(results, args.output)

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

    import matplotlib.pyplot as plt

    for algorithm in POLICIES:
        data = sorted(
            (row for row in results if row["algorithm"] == algorithm),
            key=lambda row: int(row["episode"]),
        )
        steps = [int(row["steps"]) for row in data]
        rewards = [float(row["total_reward"]) for row in data]
        plt.plot(
            steps,
            rewards,
            marker="o",
            markersize=4,
            alpha=0.75,
            label=algorithm,
        )

    plt.xlabel("Số bước")
    plt.ylabel("Tổng reward")
    plt.title("Reward theo số bước")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
