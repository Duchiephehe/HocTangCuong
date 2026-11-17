"""
Module mô phỏng môi trường Robot Hút Bụi (Vacuum Cleaner GridWorld Environment)
Tuân theo chuẩn giao diện tương tác Gymnasium / OpenAI Gym.
"""

from builtins import int
from typing import Tuple, List, Dict, Any, Optional, Callable
import random
import sys

class VacuumGridEnv:
    """
    Môi trường phòng lưới (GridWorld) cho Robot hút bụi tự hành.
    Không gian trạng thái: S = (robot_x, robot_y, dust_bitmask)
    Không gian hành động: A = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
    """
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ["LÊN (UP)", "XUỐNG (DOWN)", "TRÁI (LEFT)", "PHẢI (RIGHT)"]

    def __init__(
        self,
        width: int = 5,
        height: int = 5,
        start_pos: Tuple[int, int] = (0, 0),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        dust_positions: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 100,
        stochastic_prob: float = 0.0,
        num_dust: int = 3,
    ):
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError("width phải là số nguyên dương.")
        if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
            raise ValueError("height phải là số nguyên dương.")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps <= 0:
            raise ValueError("max_steps phải là số nguyên dương.")
        if (
            not isinstance(stochastic_prob, (int, float))
            or isinstance(stochastic_prob, bool)
            or not 0.0 <= stochastic_prob <= 1.0
        ):
            raise ValueError("stochastic_prob phải nằm trong khoảng [0, 1].")

        self.width = width
        self.height = height
        self.start_pos = self._validate_position(start_pos, "start_pos")
        obstacle_positions = (
            obstacles
            if obstacles is not None
            else [(1, 1), (2, 2), (3, 1), (3, 3)]
        )
        self.obstacles = {
            self._validate_position(pos, "obstacles")
            for pos in obstacle_positions
        }
        max_obstacles = max(1, (self.width * self.height) // 2)
        if len(self.obstacles) > max_obstacles:
            raise ValueError(
                f"Số vật cản tối đa cho lưới {self.width}×{self.height} là {max_obstacles}."
            )
        if self.start_pos in self.obstacles:
            raise ValueError("Vị trí xuất phát không được trùng vật cản.")
        if not isinstance(num_dust, int) or isinstance(num_dust, bool) or num_dust <= 0:
            raise ValueError("num_dust phải là số nguyên dương.")
        available_dust_positions = width * height - len(self.obstacles) - 1
        if num_dust > available_dust_positions:
            raise ValueError(
                f"num_dust không được lớn hơn {available_dust_positions} "
                "ô trống ngoài vị trí xuất phát."
            )

        self.num_dust = num_dust
        self.randomize_dust = dust_positions is None
        self.initial_dust = (
            self._sample_dust_positions()
            if self.randomize_dust
            else [
                self._validate_position(pos, "dust_positions")
                for pos in (dust_positions if dust_positions is not None else [])
            ]
        )
        if (
            len(self.initial_dust) != self.num_dust
            or len(set(self.initial_dust)) != self.num_dust
        ):
            raise ValueError(
                f"Cần cấu hình đúng {self.num_dust} vị trí bụi khác nhau."
            )
        if any(pos == self.start_pos or pos in self.obstacles for pos in self.initial_dust):
            raise ValueError("Vị trí bụi không được trùng robot xuất phát hoặc vật cản.")
        self.max_steps = max_steps
        self.stochastic_prob = stochastic_prob

        # Trạng thái động
        self.robot_pos = self.start_pos
        self.active_dust = set(self.initial_dust)
        self.current_step = 0
        self._episode_done = False
        self._episode_successful = False

    def _validate_position(
        self, position: Tuple[int, int], name: str
    ) -> Tuple[int, int]:
        """Kiểm tra tọa độ nằm trong lưới trước khi dùng làm vị trí."""
        if (
            not isinstance(position, (tuple, list))
            or len(position) != 2
            or any(
                not isinstance(coordinate, int) or isinstance(coordinate, bool)
                for coordinate in position
            )
        ):
            raise ValueError(f"{name} phải là tọa độ gồm hai số nguyên.")
        x, y = position
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Tọa độ {position} trong {name} nằm ngoài lưới.")
        return x, y

    def _get_dust_bitmask(self) -> int:
        """Mã hóa bụi theo chỉ số ô lưới để trạng thái nhất quán giữa các episode."""
        bitmask = 0
        for x, y in self.active_dust:
            bitmask |= 1 << (y * self.width + x)
        return bitmask

    def _sample_dust_positions(self) -> List[Tuple[int, int]]:
        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.start_pos and (x, y) not in self.obstacles
        ]
        if len(available_positions) < self.num_dust:
            raise ValueError(
                f"Cần ít nhất {self.num_dust} ô trống ngoài vị trí xuất phát "
                "để đặt bụi."
            )
        return random.sample(available_positions, self.num_dust)

    def max_obstacles(self) -> int:
        """Số vật cản tối đa theo nguyên tắc n²/2."""
        return max(1, (self.width * self.height) // 2)

    def set_num_dust(self, num_dust: int) -> None:
        """Choose a new dust count and start a fresh episode."""
        max_dust = self.width * self.height - len(self.obstacles) - 1
        if (
            not isinstance(num_dust, int)
            or isinstance(num_dust, bool)
            or not 1 <= num_dust <= max_dust
        ):
            raise ValueError(f"num_dust phải nằm trong khoảng [1, {max_dust}].")

        self.num_dust = num_dust
        self.randomize_dust = True
        self.initial_dust = self._sample_dust_positions()
        self.robot_pos = self.start_pos
        self.active_dust = set(self.initial_dust)
        self.current_step = 0
        self._episode_done = False
        self._episode_successful = False

    def set_num_obstacles(self, num_obstacles: int) -> None:
        """Thay đổi số lượng vật cản và khởi động lại episode mới."""
        max_allowed = self.max_obstacles()
        if (
            not isinstance(num_obstacles, int)
            or isinstance(num_obstacles, bool)
            or not 0 <= num_obstacles <= max_allowed
        ):
            raise ValueError(f"num_obstacles phải nằm trong khoảng [0, {max_allowed}].")

        available_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) != self.start_pos
        ]
        if len(available_positions) < num_obstacles:
            raise ValueError(
                f"Không đủ ô trống để đặt {num_obstacles} vật cản trên lưới {self.width}×{self.height}."
            )

        self.obstacles = set(random.sample(available_positions, num_obstacles))
        self.num_dust = min(self.num_dust, self.width * self.height - len(self.obstacles) - 1)
        self.randomize_dust = True
        self.initial_dust = self._sample_dust_positions()
        self.robot_pos = self.start_pos
        self.active_dust = set(self.initial_dust)
        self.current_step = 0
        self._episode_done = False
        self._episode_successful = False

    def set_grid_size(self, size: int) -> None:
        """Resize the environment to a square grid and start a fresh episode."""
        if not isinstance(size, int) or isinstance(size, bool) or size < 4:
            raise ValueError("Kích thước lưới vuông phải là số nguyên từ 4 trở lên.")
        if self.start_pos[0] >= size or self.start_pos[1] >= size:
            raise ValueError("Kích thước mới không chứa được vị trí xuất phát.")

        self.width = size
        self.height = size
        max_allowed = self.max_obstacles()
        target_obstacles = min(len(self.obstacles), max_allowed)
        if target_obstacles == 0:
            self.obstacles = set()
        else:
            self.set_num_obstacles(target_obstacles)
        max_dust = size * size - len(self.obstacles) - 1
        self.set_num_dust(min(self.num_dust, max_dust))

    def get_state(self) -> Tuple[int, int, int]:
        """Trả về tuple trạng thái s = (x, y, dust_bitmask)"""
        return (self.robot_pos[0], self.robot_pos[1], self._get_dust_bitmask())

    def reset(self) -> Tuple[Tuple[int, int, int], Dict[str, Any]]:
        """Đặt lại episode; chỉ sinh vị trí bụi mới nếu episode trước đã dọn sạch."""
        if self.randomize_dust and self._episode_successful:
            self.initial_dust = self._sample_dust_positions()
        self.robot_pos = self.start_pos
        self.active_dust = set(self.initial_dust)
        self.current_step = 0
        self._episode_done = False
        self._episode_successful = False
        info = {
            "initial_dust_count": len(self.initial_dust),
            "dust_remaining": len(self.active_dust),
            "dust_positions": list(self.initial_dust),
        }
        return self.get_state(), info

    def step(self, action: int) -> Tuple[Tuple[int, int, int], float, bool, bool, Dict[str, Any]]:
        """
        Thực thi 1 hành động của robot.
        Trả về: (next_state, reward, terminated, truncated, info)
        """
        if self._episode_done:
            raise RuntimeError("Episode đã kết thúc; hãy gọi reset() trước khi step() tiếp.")
        if (
            not isinstance(action, int)
            or isinstance(action, bool)
            or action
            not in (
                self.ACTION_UP,
                self.ACTION_DOWN,
                self.ACTION_LEFT,
                self.ACTION_RIGHT,
            )
        ):
            raise ValueError(f"Hành động không hợp lệ: {action}. Chọn một giá trị từ 0 đến 3.")

        self.current_step += 1

        # Xử lý trượt ngẫu nhiên nếu là môi trường stochastic
        effective_action = action
        if self.stochastic_prob > 0.0 and random.random() < self.stochastic_prob:
            effective_action = random.choice([0, 1, 2, 3])

        # Tính tọa độ dự kiến
        dx, dy = 0, 0
        if effective_action == self.ACTION_UP:
            dy = 1
        elif effective_action == self.ACTION_DOWN:
            dy = -1
        elif effective_action == self.ACTION_LEFT:
            dx = -1
        elif effective_action == self.ACTION_RIGHT:
            dx = 1

        target_x = self.robot_pos[0] + dx
        target_y = self.robot_pos[1] + dy

        reward = 0.0
        collision = False

        # Kiểm tra va chạm biên hoặc vật cản
        if (
            target_x < 0
            or target_x >= self.width
            or target_y < 0
            or target_y >= self.height
            or (target_x, target_y) in self.obstacles
        ):
            # Va chạm: phạt và giữ nguyên vị trí.
            reward += -20.0
            collision = True
        else:
            # Di chuyển hợp lệ
            self.robot_pos = (target_x, target_y)
            reward += -1.0  # Chi phí tiêu hao năng lượng mỗi bước di chuyển

        # Kiểm tra hút bụi
        cleaned_dust = False
        if self.robot_pos in self.active_dust:
            self.active_dust.remove(self.robot_pos)
            reward += 20.0  # Thưởng hút sạch 1 vị trí bụi
            cleaned_dust = True

        # Kiểm tra điều kiện kết thúc
        terminated = len(self.active_dust) == 0
        if terminated:
            reward += 100.0  # Thưởng hoàn thành dọn sạch căn phòng

        # Kiểm tra hết pin (hết số bước tối đa)
        truncated = self.current_step >= self.max_steps and not terminated
        self._episode_done = terminated or truncated
        self._episode_successful = terminated

        info = {
            "step": self.current_step,
            "collision": collision,
            "cleaned_dust": cleaned_dust,
            "dust_remaining": len(self.active_dust),
        }

        return self.get_state(), reward, terminated, truncated, info

    def render_ascii(self) -> str:
        """Hiển thị bản đồ căn phòng dạng ASCII trên terminal"""
        lines = []
        header = f"--- Bước: {self.current_step} | Bụi còn: {len(self.active_dust)} ---"
        lines.append(header)
        for y in range(self.height - 1, -1, -1):
            row_str = ""
            for x in range(self.width):
                pos = (x, y)
                if pos == self.robot_pos:
                    row_str += " [R] "  # Robot
                elif pos in self.obstacles:
                    row_str += "  #  "  # Vật cản
                elif pos in self.active_dust:
                    row_str += "  *  "  # Bụi
                else:
                    row_str += "  .  "  # Ô trống
            lines.append(row_str)
        return "\n".join(lines)

    def _draw_matplotlib(self, ax: Any) -> None:
        """Vẽ trạng thái hiện tại lên một Matplotlib axes."""
        from matplotlib.lines import Line2D # type: ignore
        from matplotlib.patches import Patch, Rectangle # pyright: ignore[reportMissingModuleSource]

        ax.clear()
        for x, y in self.initial_dust:
            if (x, y) not in self.active_dust:
                ax.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="#E8F5E9",
                        edgecolor="none",
                    )
                )
        for x, y in self.obstacles:
            ax.add_patch(
                Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor="#59636E",
                    edgecolor="none",
                )
            )

        if self.active_dust:
            dust_x, dust_y = zip(*self.active_dust)
            ax.scatter(
                dust_x,
                dust_y,
                marker="*",
                s=300,
                color="#F4B400",
                edgecolors="#A66B00",
                linewidths=0.8,
                zorder=3,
            )

        robot_x, robot_y = self.robot_pos
        ax.scatter(
            [robot_x],
            [robot_y],
            marker="o",
            s=420,
            color="#2878B5",
            edgecolors="white",
            linewidths=2,
            zorder=4,
        )
        ax.text(
            robot_x,
            robot_y,
            "R",
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
            zorder=5,
        )

        ax.set_xlim(-0.5, self.width - 0.5)
        ax.set_ylim(-0.5, self.height - 0.5)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.set_xticks([x - 0.5 for x in range(self.width + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(self.height + 1)], minor=True)
        ax.grid(which="minor", color="#CBD5E0", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_aspect("equal")
        ax.set_xlabel("Cột (x)")
        ax.set_ylabel("Hàng (y)")
        ax.set_title(
            f"Robot hút bụi | Bước: {self.current_step}/{self.max_steps} "
            f"| Bụi còn: {len(self.active_dust)}"
        )
        ax.legend(
            handles=[
                Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="#2878B5",
                    markeredgecolor="white", markersize=11, label="Robot",
                ),
                Line2D(
                    [0], [0], marker="*", color="w", markerfacecolor="#F4B400",
                    markeredgecolor="#A66B00", markersize=13, label="Bụi",
                ),
                Patch(facecolor="#59636E", label="Vật cản"),
                Patch(facecolor="#E8F5E9", label="Ô đã dọn"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=4,
            frameon=False,
        )

    def render_matplotlib(
        self, show: bool = True, axes: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """Hiển thị bản đồ; khi tạo cửa sổ mới có thanh trượt chọn số bụi."""
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")

        interactive = axes is None
        if axes is None:
            fig, ax = plt.subplots(figsize=(max(5, self.width), max(4, self.height)))
        else:
            ax = axes
            fig = ax.figure
        self._draw_matplotlib(ax)
        if interactive:
            from matplotlib.widgets import Slider

            fig.subplots_adjust(bottom=0.44)
            size_slider_ax = fig.add_axes([0.2, 0.26, 0.6, 0.04])
            size_slider = Slider(
                size_slider_ax,
                "Kích thước lưới n×n",
                4,
                20,
                valinit=self.width,
                valstep=1,
            )

            obstacle_slider_ax = fig.add_axes([0.2, 0.16, 0.6, 0.04])
            obstacle_slider = Slider(
                obstacle_slider_ax,
                "Số vật cản",
                0,
                self.max_obstacles(),
                valinit=len(self.obstacles),
                valstep=1,
            )

            dust_slider_ax = fig.add_axes([0.2, 0.06, 0.6, 0.04])
            dust_slider = Slider(
                dust_slider_ax,
                "Số vị trí bụi",
                1,
                self.width * self.height - len(self.obstacles) - 1,
                valinit=self.num_dust,
                valstep=1,
            )

            def update_grid_size(value: float) -> None:
                self.set_grid_size(int(value))
                obstacle_slider.valmax = self.max_obstacles()
                obstacle_slider.ax.set_xlim(obstacle_slider.valmin, obstacle_slider.valmax)
                if int(obstacle_slider.val) != len(self.obstacles):
                    obstacle_slider.set_val(len(self.obstacles))
                max_dust = self.width * self.height - len(self.obstacles) - 1
                dust_slider.valmax = max_dust
                dust_slider.ax.set_xlim(dust_slider.valmin, max_dust)
                if int(dust_slider.val) > max_dust:
                    dust_slider.set_val(max_dust)
                elif int(dust_slider.val) < 1:
                    dust_slider.set_val(1)
                self._draw_matplotlib(ax)
                fig.canvas.draw_idle()

            def update_obstacle_count(value: float) -> None:
                self.set_num_obstacles(int(value))
                max_dust = self.width * self.height - len(self.obstacles) - 1
                dust_slider.valmax = max_dust
                dust_slider.ax.set_xlim(dust_slider.valmin, max_dust)
                if int(dust_slider.val) > max_dust:
                    dust_slider.set_val(max_dust)
                elif int(dust_slider.val) < 1:
                    dust_slider.set_val(1)
                self._draw_matplotlib(ax)
                fig.canvas.draw_idle()

            def update_dust_count(value: float) -> None:
                self.set_num_dust(int(value))
                self._draw_matplotlib(ax)
                fig.canvas.draw_idle()

            size_slider.on_changed(update_grid_size)
            obstacle_slider.on_changed(update_obstacle_count)
            dust_slider.on_changed(update_dust_count)
            setattr(fig, "_grid_size_slider", size_slider)
            setattr(fig, "_obstacle_count_slider", obstacle_slider)
            setattr(fig, "_dust_count_slider", dust_slider)
        else:
            fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def animate_matplotlib(
        self,
        policy: Callable[[Tuple[int, int, int]], int],
        interval_ms: int = 500,
        reset: bool = True,
    ) -> None:
        """Chạy và hiển thị chính sách của agent theo từng bước bằng Matplotlib."""
        import importlib

        plt = importlib.import_module("matplotlib.pyplot")
        from matplotlib.animation import FuncAnimation # type: ignore

        if (
            not isinstance(interval_ms, int)
            or isinstance(interval_ms, bool)
            or interval_ms <= 0
        ):
            raise ValueError("interval_ms phải là số nguyên dương.")
        if reset:
            self.reset()
        elif self._episode_done:
            raise RuntimeError("Episode đã kết thúc; hãy gọi reset() trước khi chạy hoạt ảnh.")

        fig, ax = plt.subplots(figsize=(max(5, self.width), max(4, self.height)))
        self._draw_matplotlib(ax)
        episode_number = 1
        fig.suptitle(f"Lượt dọn: {episode_number}")
        animation = None

        def update(_frame: int) -> None:
            nonlocal episode_number
            if self._episode_done:
                if not self._episode_successful:
                    if animation is not None and animation.event_source is not None:
                        animation.event_source.stop()
                    return
                self.reset()
                episode_number += 1
                fig.suptitle(f"Lượt dọn: {episode_number}")
                self._draw_matplotlib(ax)
                fig.tight_layout()
                return
            action = policy(self.get_state())
            self.step(action)
            self._draw_matplotlib(ax)
            fig.tight_layout()

        def initialize() -> Tuple[Any, ...]:
            return ()

        animation = FuncAnimation(
            fig,
            update,
            init_func=initialize,
            frames=None,
            interval=interval_ms,
            repeat=False,
            cache_frame_data=False,
        )
        plt.show()


if __name__ == "__main__":
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    env = VacuumGridEnv(start_pos=(0, 0))
    env.render_matplotlib()
