"""Run the nearest-dust BFS policy in the interactive grid window."""

import argparse
import importlib
from typing import Any, Optional

from compare_algorithms import nearest_dust_policy
from vacuum_env import VacuumGridEnv

MIN_GRID_SIZE = 4
MAX_GRID_SIZE = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Độ trễ giữa các bước tính bằng mili giây (mặc định: 500).",
    )
    args = parser.parse_args()
    if args.interval_ms <= 0:
        parser.error("--interval-ms phải là số nguyên dương.")

    plt = importlib.import_module("matplotlib.pyplot")
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider

    env = VacuumGridEnv()
    fig, ax = env.render_matplotlib(show=False)
    fig.subplots_adjust(bottom=0.44)
    size_slider = fig._grid_size_slider
    obstacle_slider = fig._obstacle_count_slider
    dust_slider = fig._dust_count_slider
    fig.suptitle("BFS: đi tới vị trí bụi gần nhất")
    animation: Optional[Any] = None

    def restart(_value: float) -> None:
        fig.suptitle(f"BFS trên lưới {env.width}×{env.height}")
        if animation is not None and animation.event_source is not None:
            animation.event_source.start()
        fig.canvas.draw_idle()

    def update_run(_value: float) -> None:
        env.max_steps = max(100, env.width * env.height * env.num_dust)
        restart(_value)

    def change_grid_size(value: float) -> None:
        env.set_grid_size(int(value))
        max_obstacles = env.max_obstacles()
        obstacle_slider.valmax = max_obstacles
        obstacle_slider.ax.set_xlim(obstacle_slider.valmin, max_obstacles)
        if int(obstacle_slider.val) != len(env.obstacles):
            obstacle_slider.set_val(len(env.obstacles))
        else:
            env._draw_matplotlib(ax)
            fig.canvas.draw_idle()

        max_dust = env.width * env.height - len(env.obstacles) - 1
        dust_slider.valmax = max_dust
        dust_slider.ax.set_xlim(dust_slider.valmin, max_dust)
        if int(dust_slider.val) != env.num_dust:
            dust_slider.set_val(env.num_dust)
        else:
            env._draw_matplotlib(ax)
            fig.canvas.draw_idle()
        update_run(value)

    size_slider.on_changed(change_grid_size)
    obstacle_slider.on_changed(lambda value: update_run(value))
    dust_slider.on_changed(update_run)

    def update(_frame: int) -> None:
        if env._episode_done:
            if animation is not None and animation.event_source is not None:
                animation.event_source.stop()
            fig.suptitle(
                f"BFS hoàn thành trên lưới {env.width}×{env.height} "
                "| Chọn kích thước hoặc số bụi để chạy lại"
            )
            return

        env.step(nearest_dust_policy(env))
        env._draw_matplotlib(ax)
        if env._episode_done:
            fig.suptitle(
                f"BFS hoàn thành trên lưới {env.width}×{env.height} "
                f"| Số bước: {env.current_step}"
            )
        fig.canvas.draw_idle()

    animation = FuncAnimation(
        fig,
        update,
        interval=args.interval_ms,
        repeat=False,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
