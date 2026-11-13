"""Train and run a Q-Learning agent in the vacuum cleaner grid environment."""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vacuum_env import VacuumGridEnv

State = Tuple[int, int, int]
QTable = Dict[State, List[float]]


def _new_episode_env(template: VacuumGridEnv, rng: random.Random) -> VacuumGridEnv:
    """Create an episode with the same configuration and a fresh dust layout."""
    if template.randomize_dust:
        available = [
            (x, y)
            for y in range(template.height)
            for x in range(template.width)
            if (x, y) != template.start_pos and (x, y) not in template.obstacles
        ]
        dust_positions = rng.sample(available, 3)
    else:
        dust_positions = list(template.initial_dust)

    return VacuumGridEnv(
        width=template.width,
        height=template.height,
        start_pos=template.start_pos,
        obstacles=list(template.obstacles),
        dust_positions=dust_positions,
        max_steps=template.max_steps,
        stochastic_prob=template.stochastic_prob,
    )


def _values_for(q_table: QTable, state: State) -> List[float]:
    return q_table.setdefault(state, [0.0] * len(VacuumGridEnv.ACTION_NAMES))


def _choose_action(
    q_table: QTable, state: State, epsilon: float, rng: random.Random
) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(VacuumGridEnv.ACTION_NAMES))

    values = _values_for(q_table, state)
    best_value = max(values)
    best_actions = [action for action, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def train_q_learning(
    env: VacuumGridEnv,
    episodes: int = 600,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[QTable, List[float]]:
    """Train a tabular Q-Learning agent and return its Q-table and episode rewards."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes <= 0:
        raise ValueError("episodes phải là số nguyên dương.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha phải nằm trong khoảng (0, 1].")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma phải nằm trong khoảng [0, 1].")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon phải nằm trong khoảng [0, 1].")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay phải nằm trong khoảng (0, 1].")
    if not 0.0 <= min_epsilon <= 1.0:
        raise ValueError("min_epsilon phải nằm trong khoảng [0, 1].")

    rng = random.Random(seed)
    q_table: QTable = {}
    reward_history: List[float] = []
    current_epsilon = epsilon

    for _ in range(episodes):
        episode_env = _new_episode_env(env, rng)
        state, _ = episode_env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = _choose_action(q_table, state, current_epsilon, rng)
            next_state, reward, terminated, truncated, _ = episode_env.step(action)

            values = _values_for(q_table, state)
            target = reward
            if not terminated:
                target += gamma * max(_values_for(q_table, next_state))
            values[action] += alpha * (target - values[action])

            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

    return q_table, reward_history


def save_training_plot(reward_history: List[float], output: Path) -> None:
    """Save episode rewards and their 100-episode moving average as a chart."""
    if not reward_history:
        raise ValueError("reward_history không được để trống.")

    import matplotlib.pyplot as plt

    window = min(100, len(reward_history))
    moving_average = [
        sum(reward_history[index - window + 1 : index + 1]) / window
        for index in range(window - 1, len(reward_history))
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        range(1, len(reward_history) + 1),
        reward_history,
        color="#9CA3AF",
        alpha=0.3,
        linewidth=0.7,
        label="Reward mỗi episode",
    )
    axis.plot(
        range(window, len(reward_history) + 1),
        moving_average,
        color="#1565C0",
        linewidth=2,
        label=f"Trung bình trượt ({window} episode)",
    )
    axis.set_xlabel("Episode")
    axis.set_ylabel("Tổng reward")
    axis.set_title("Q-Learning: reward theo episode")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run_greedy_episode(
    env: VacuumGridEnv, q_table: QTable, seed: Optional[int] = None
) -> Tuple[float, int, int, bool]:
    """Run one greedy episode, printing the map after each robot action."""
    rng = random.Random(seed)
    state, info = env.reset()
    total_reward = 0.0
    collisions = 0
    terminated = False
    truncated = False

    print(f"Bụi ban đầu: {info['dust_positions']}")
    print(env.render_ascii())

    while not (terminated or truncated):
        values = _values_for(q_table, state)
        best_value = max(values)
        best_actions = [
            action for action, value in enumerate(values) if value == best_value
        ]
        action = rng.choice(best_actions)
        next_state, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        collisions += int(step_info["collision"])

        print(
            f"Bước {step_info['step']}: {VacuumGridEnv.ACTION_NAMES[action]} "
            f"| thưởng {reward:g} | va chạm {step_info['collision']} "
            f"| bụi còn {step_info['dust_remaining']}"
        )
        print(env.render_ascii())
        state = next_state

    return total_reward, env.current_step, collisions, terminated


def animate_greedy_episode(
    env: VacuumGridEnv,
    q_table: QTable,
    interval_ms: int = 500,
    seed: Optional[int] = None,
) -> Tuple[float, int, int, bool]:
    """Animate one greedy episode on the environment's Matplotlib map."""
    if not isinstance(interval_ms, int) or isinstance(interval_ms, bool) or interval_ms <= 0:
        raise ValueError("interval_ms phải là số nguyên dương.")

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    from matplotlib.animation import FuncAnimation

    state, info = env.reset()
    total_reward = 0.0
    collisions = 0
    terminated = False
    truncated = False
    rng = random.Random(seed)
    fig, ax = env.render_matplotlib(show=False)
    fig.suptitle(f"Q-Learning | Bụi ban đầu: {info['dust_positions']}")
    animation = None

    def update(_frame: int) -> None:
        nonlocal state, total_reward, collisions, terminated, truncated
        if terminated or truncated:
            if animation is not None and animation.event_source is not None:
                animation.event_source.stop()
            return

        values = _values_for(q_table, state)
        best_value = max(values)
        best_actions = [i for i, value in enumerate(values) if value == best_value]
        action = rng.choice(best_actions)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        collisions += int(info["collision"])
        env.render_matplotlib(show=False, axes=ax)
        status = "Đã dọn sạch" if terminated else (
            "Hết số bước" if truncated else VacuumGridEnv.ACTION_NAMES[action]
        )
        fig.suptitle(
            f"Q-Learning | Bước: {info['step']} | {status} "
            f"| Bụi còn: {info['dust_remaining']}"
        )
        if terminated or truncated:
            if animation is not None and animation.event_source is not None:
                animation.event_source.stop()

    animation = FuncAnimation(
        fig,
        update,
        frames=None,
        interval=interval_ms,
        repeat=False,
        cache_frame_data=False,
    )
    plt.show()

    successful = terminated
    status = "dọn sạch thành công" if successful else "hết số bước tối đa"
    print(
        f"Kết quả: {status}; số bước={env.current_step}; "
        f"va chạm={collisions}; tổng thưởng={total_reward:g}."
    )
    return total_reward, env.current_step, collisions, successful


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Huấn luyện và chạy Q-Learning cho robot hút bụi."
    )
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("q_learning_training.png"),
        help="Đường dẫn lưu biểu đồ reward theo episode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render",
        choices=("matplotlib", "ascii"),
        default="matplotlib",
        help="Hiển thị robot bằng cửa sổ đồ họa hoặc bản đồ văn bản.",
    )
    parser.add_argument("--interval-ms", type=int, default=500)
    args = parser.parse_args()

    env = VacuumGridEnv()
    print(f"Bố cục bụi ngẫu nhiên của lượt kiểm thử: {env.initial_dust}")
    q_table, reward_history = train_q_learning(
        env,
        episodes=args.episodes,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.05,
        seed=args.seed,
    )

    recent_count = min(20, len(reward_history))
    recent_average = sum(reward_history[-recent_count:]) / recent_count
    save_training_plot(reward_history, args.plot_output)
    print(
        f"Huấn luyện xong {len(reward_history)} episode; "
        f"{len(q_table)} trạng thái đã học; "
        f"thưởng trung bình {recent_count} episode cuối: {recent_average:.2f}.\n"
        f"Biểu đồ reward đã lưu tại: {args.plot_output}"
    )
    print("\nChạy chính sách đã học:")
    if args.render == "matplotlib":
        animate_greedy_episode(
            env, q_table, interval_ms=args.interval_ms, seed=args.seed
        )
    else:
        result = run_greedy_episode(env, q_table, seed=args.seed)
        total_reward, steps, collisions, successful = result
        status = "dọn sạch thành công" if successful else "hết số bước tối đa"
        print(
            f"\nKết quả: {status}; số bước={steps}; va chạm={collisions}; "
            f"tổng thưởng={total_reward:g}."
        )


if __name__ == "__main__":
    main()
