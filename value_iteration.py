"""
=============================================================================
ĐỀ TÀI: TỐI ƯU HÓA ĐƯỜNG ĐI CHO ROBOT HÚT BỤI BẰNG HỌC TĂNG CƯỜNG
Môn học: Học tăng cường và ứng dụng
Nhóm 5 - 65KTPM (Báo cáo TH1: Mô hình hóa bài toán với MDP)

Thuật toán: VALUE ITERATION (Quy hoạch động - Dynamic Programming, Model-based)
Dựa theo thiết kế và mô hình hóa trong tài liệu báo cáo:
- Không gian trạng thái: S = (robot_x, robot_y, dust_bitmask)
- Không gian hành động: A = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}
- Phần thưởng:
    + Di chuyển hợp lệ: -1.0 (chi phí tiêu hao năng lượng mỗi bước)
    + Va chạm vật cản / biên: -20.0 (phạt va chạm, giữ nguyên vị trí)
    + Hút sạch 1 vị trí bụi: +20.0
    + Hoàn thành dọn sạch toàn bộ bụi: +100.0 (trạng thái kết thúc)
- Phương trình Bellman tối ưu:
    V_{k+1}(s) = max_a [ R(s, a, T(s,a)) + gamma * V_k(T(s,a)) ]
- Chính sách tối ưu:
    pi*(s) = argmax_a [ R(s, a, T(s,a)) + gamma * V*(T(s,a)) ]
=============================================================================
"""

import sys
from typing import Dict, List, Tuple, Optional
from vacuum_env import VacuumGridEnv


class ValueIterationAgent:
    """
    Thuật toán Value Iteration tìm hàm giá trị tối ưu V* và chính sách tối ưu pi*
    dựa trên mô hình chuyển trạng thái và hàm phần thưởng đã biết trước.
    """

    # Ánh xạ hành động sang độ dịch chuyển tọa độ (dx, dy) theo vacuum_env.py
    # 0: UP (y + 1), 1: DOWN (y - 1), 2: LEFT (x - 1), 3: RIGHT (x + 1)
    ACTION_DELTAS = {
        VacuumGridEnv.ACTION_UP: (0, 1),
        VacuumGridEnv.ACTION_DOWN: (0, -1),
        VacuumGridEnv.ACTION_LEFT: (-1, 0),
        VacuumGridEnv.ACTION_RIGHT: (1, 0),
    }

    def __init__(
        self,
        env: VacuumGridEnv,
        gamma: float = 0.9,
        theta: float = 1e-6,
    ):
        """
        Khởi tạo Agent Value Iteration.

        Tham số:
            env: Thể hiện của môi trường VacuumGridEnv
            gamma: Hệ số chiết khấu (Discount factor, mặc định 0.9 theo báo cáo)
            theta: Ngưỡng hội tụ sai số Bellman (mặc định 1e-6 theo báo cáo)
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Lấy thông tin bản đồ từ môi trường
        self.width = env.width
        self.height = env.height
        self.obstacles = set(env.obstacles)
        self.dust_positions = list(env.initial_dust)

        # 4 hành động hợp lệ
        self.actions = [
            VacuumGridEnv.ACTION_UP,
            VacuumGridEnv.ACTION_DOWN,
            VacuumGridEnv.ACTION_LEFT,
            VacuumGridEnv.ACTION_RIGHT,
        ]

        # Khởi tạo tập trạng thái, bảng giá trị V và bảng chính sách pi
        self.states: List[Tuple[int, int, int]] = self._generate_state_space()
        self.V: Dict[Tuple[int, int, int], float] = {s: 0.0 for s in self.states}
        self.policy: Dict[Tuple[int, int, int], int] = {}

    def _generate_state_space(self) -> List[Tuple[int, int, int]]:
        """
        Sinh không gian trạng thái S = (x, y, dust_bitmask).
        Các vị trí (x, y) hợp lệ là các ô không phải vật cản (21 ô).
        dust_bitmask là tổ hợp các ô bụi còn lại (2^3 = 8 trạng thái bụi).
        """
        # 1. Tập hợp các ô robot có thể đứng (21 ô hợp lệ trên lưới 5x5 có 4 vật cản)
        valid_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self.obstacles
        ]

        # 2. Tập hợp các bitmask khả dĩ từ các vị trí bụi ban đầu
        # Với 3 vị trí bụi, có 2^3 = 8 tập con (từ còn nguyên 3 bụi đến sạch 0 bụi)
        dust_masks = []
        n_dust = len(self.dust_positions)
        for i in range(1 << n_dust):
            mask = 0
            for bit_idx in range(n_dust):
                if (i >> bit_idx) & 1:
                    px, py = self.dust_positions[bit_idx]
                    mask |= (1 << (py * self.width + px))
            dust_masks.append(mask)

        # 3. Kết hợp (x, y) và dust_bitmask để tạo các trạng thái
        states = []
        for x, y in valid_cells:
            for mask in dust_masks:
                states.append((x, y, mask))

        return states

    def transition_model(
        self, state: Tuple[int, int, int], action: int
    ) -> Tuple[Tuple[int, int, int], float]:
        """
        Mô hình chuyển trạng thái xác định T(s, a) -> (s', r).
        Tuân thủ đúng quy tắc chuyển trạng thái và hàm phần thưởng trong báo cáo TH1:
        - Va chạm tường / vật cản: giữ nguyên vị trí, phạt -20.
        - Bước đi hợp lệ: phạt -1 (chi phí năng lượng).
        - Đến ô có bụi: dọn sạch ô đó (tắt bit), thưởng +20.
        - Dọn sạch toàn bộ bụi (mask == 0): thưởng hoàn thành +100.
        """
        x, y, mask = state
        dx, dy = self.ACTION_DELTAS[action]
        next_x, next_y = x + dx, y + dy

        # 1. Kiểm tra va chạm biên hoặc vật cản
        is_collision = (
            next_x < 0
            or next_x >= self.width
            or next_y < 0
            or next_y >= self.height
            or (next_x, next_y) in self.obstacles
        )

        if is_collision:
            # Va chạm: giữ nguyên tọa độ và bitmask, phạt -20
            next_state = (x, y, mask)
            reward = -20.0
            return next_state, reward

        # 2. Di chuyển thành công đến ô hợp lệ
        reward = -1.0
        next_mask = mask

        # Kiểm tra xem ô đích có bụi hay không
        target_bit = 1 << (next_y * self.width + next_x)
        if mask & target_bit:
            # Dọn sạch bụi ở ô đích: tắt bit tương ứng và cộng thưởng
            next_mask = mask ^ target_bit
            reward += 20.0

            # Nếu đã dọn sạch toàn bộ bụi (trạng thái kết thúc): cộng thưởng lớn
            if next_mask == 0:
                reward += 100.0

        next_state = (next_x, next_y, next_mask)
        return next_state, reward

    def train(self) -> int:
        """
        Thực thi thuật toán Value Iteration theo mã giả trong báo cáo:
        1. Lặp cập nhật giá trị trạng thái V(s) qua phương trình Bellman tối ưu.
        2. Kiểm tra điều kiện hội tụ: delta = max |V_new(s) - V_old(s)| < theta.
        3. Trích xuất chính sách tối ưu pi*(s).

        Trả về:
            Số vòng lặp cần thiết để thuật toán hội tụ.
        """
        iteration = 0

        # Vòng lặp cập nhật giá trị
        while True:
            iteration += 1
            delta = 0.0

            # Bản sao lưu trữ giá trị mới ở vòng lặp hiện tại
            V_new = self.V.copy()

            for state in self.states:
                _, _, mask = state

                # Trạng thái kết thúc (mask == 0, đã dọn sạch bụi): giữ V(s) = 0
                if mask == 0:
                    V_new[state] = 0.0
                    continue

                # Tính Q(s, a) cho tất cả các hành động
                q_values = []
                for action in self.actions:
                    next_state, reward = self.transition_model(state, action)
                    q = reward + self.gamma * self.V[next_state]
                    q_values.append(q)

                # Giá trị trạng thái tối ưu theo Bellman: V(s) = max_a Q(s, a)
                best_value = max(q_values)
                delta = max(delta, abs(best_value - self.V[state]))
                V_new[state] = best_value

            self.V = V_new

            # Kiểm tra điều kiện dừng
            if delta < self.theta:
                break

        # Trích xuất chính sách tối ưu pi*(s) = argmax_a Q(s, a)
        for state in self.states:
            _, _, mask = state
            if mask == 0:
                continue

            best_action = self.actions[0]
            best_q = -float("inf")
            for action in self.actions:
                next_state, reward = self.transition_model(state, action)
                q = reward + self.gamma * self.V[next_state]
                if q > best_q:
                    best_q = q
                    best_action = action

            self.policy[state] = best_action

        return iteration

    def get_action(self, state: Tuple[int, int, int]) -> int:
        """Trả về hành động tối ưu theo chính sách đã học."""
        return self.policy.get(state, VacuumGridEnv.ACTION_UP)

    def run_simulation(self, max_steps: int = 50, verbose: bool = True) -> Dict[str, float]:
        """
        Chạy mô phỏng kiểm thử robot với chính sách tối ưu đã tìm được.
        In giao diện ASCII từng bước để theo dõi trực quan đường đi.
        """
        state, info = self.env.reset()
        total_reward = 0.0
        total_steps = 0
        collision_count = 0

        if verbose:
            print("\n" + "=" * 55)
            print(" BẮT ĐẦU MÔ PHỎNG VẬN HÀNH ROBOT THEO CHÍNH SÁCH VALUE ITERATION")
            print("=" * 55)
            print("Trạng thái khởi đầu:")
            print(self.env.render_ascii())

        done = False
        while not done and total_steps < max_steps:
            total_steps += 1
            action = self.get_action(state)
            action_name = VacuumGridEnv.ACTION_NAMES[action]

            next_state, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            if step_info["collision"]:
                collision_count += 1

            if verbose:
                print(f"\n--- Bước {total_steps}: Hành động [{action_name}] ---")
                print(f"Phần thưởng bước: {reward:+.1f} | Tổng Reward: {total_reward:+.1f}")
                if step_info["cleaned_dust"]:
                    print("=> [VỪA HÚT ĐƯỢC BỤI!]")
                if step_info["collision"]:
                    print("=> [CẢNH BÁO: VA CHẠM!]")
                print(self.env.render_ascii())

            state = next_state
            done = terminated or truncated

        # Thống kê kết quả
        cleaned_ratio = (
            (len(self.dust_positions) - len(self.env.active_dust))
            / len(self.dust_positions)
        ) * 100.0

        stats = {
            "total_steps": total_steps,
            "total_reward": total_reward,
            "cleaned_ratio": cleaned_ratio,
            "collision_count": collision_count,
            "is_success": len(self.env.active_dust) == 0,
        }

        if verbose:
            print("\n" + "=" * 55)
            print(" TỔNG KẾT KẾT QUẢ KIỂM THỬ")
            print("=" * 55)
            print(f"- Trạng thái kết thúc: {'THÀNH CÔNG (Dọn sạch 100% bụi)' if stats['is_success'] else 'CHƯA HOÀN THÀNH'}")
            print(f"- Tổng số bước di chuyển : {stats['total_steps']}")
            print(f"- Tổng phần thưởng tích lũy: {stats['total_reward']:.1f}")
            print(f"- Tỷ lệ bụi đã dọn sạch  : {stats['cleaned_ratio']:.1f}%")
            print(f"- Số lần va chạm         : {stats['collision_count']}")
            print("=" * 55 + "\n")

        return stats


def main():
    # Cấu hình encoding UTF-8 cho terminal Windows
    if callable(getattr(sys.stdout, "reconfigure", None)):
        sys.stdout.reconfigure(encoding="utf-8")

    print("Khởi tạo môi trường mô phỏng Robot Hút Bụi (GridWorld 5x5)...")
    # Khởi tạo môi trường với các vị trí vật cản và bụi mẫu
    env = VacuumGridEnv(
        width=5,
        height=5,
        start_pos=(0, 0),
        obstacles=[(1, 1), (2, 2), (3, 1), (3, 3)],
        dust_positions=[(0, 2), (2, 4), (4, 2)],
        stochastic_prob=0.0,  # Môi trường tất định (Deterministic) theo báo cáo TH1
    )

    print(f"- Kích thước lưới: {env.width}x{env.height}")
    print(f"- Vị trí xuất phát của Robot: {env.start_pos}")
    print(f"- Vị trí 4 vật cản cố định  : {list(env.obstacles)}")
    print(f"- Vị trí 3 đống bụi ban đầu : {list(env.initial_dust)}")

    # 1. Khởi tạo Agent Value Iteration
    agent = ValueIterationAgent(env=env, gamma=0.9, theta=1e-6)
    print(f"- Tổng số trạng thái mô hình hóa: {len(agent.states)}")

    # 2. Huấn luyện tìm nghiệm tối ưu
    print("\nĐang tiến hành giải nghiệm bằng thuật toán Value Iteration...")
    num_iterations = agent.train()
    start_state = env.get_state()
    start_value = agent.V.get(start_state, 0.0)
    print(f"=> Thuật toán đã HỘI TỤ sau {num_iterations} vòng lặp!")
    print(f"=> Giá trị V* tại trạng thái xuất phát {start_state}: {start_value:.4f}")

    # 3. Chạy mô phỏng đường đi tối ưu
    # agent.run_simulation(max_steps=50, verbose=True)
    # Thay agent.run_simulation(...) bằng:
    agent.train()
    # 3. Chạy mô phỏng đường đi tối ưu bằng Matplotlib animation
    env.animate_matplotlib(policy=agent.get_action, interval_ms=500)

    # Option 1: Hiển thị 1 frame tĩnh
    env.render_matplotlib()

    # Option 2: Animation từng bước theo chính sách
    env.animate_matplotlib(policy=agent.get_action, interval_ms=500)


if __name__ == "__main__":
    main()
