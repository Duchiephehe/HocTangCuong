# Báo cáo thuật toán Q-Learning cho robot hút bụi

## 1. Mục tiêu

Q-Learning được áp dụng để robot học cách di chuyển trong phòng dạng lưới và thu gom toàn bộ bụi với tổng phần thưởng cao. Robot tương tác với môi trường, nhận phần thưởng sau mỗi hành động và lưu kinh nghiệm trong bảng Q. Thuật toán là **model-free** (không cần biết trước mô hình chuyển trạng thái) và **off-policy** (khi cập nhật dùng hành động tốt nhất ước tính ở trạng thái tiếp theo, không phụ thuộc hành động khám phá đã chọn).

Mã thuật toán nằm trong [`q_learning.py`](./q_learning.py); môi trường mô phỏng được định nghĩa trong [`vacuum_env.py`](./vacuum_env.py).

## 2. Mô hình môi trường

Môi trường mặc định là lưới 5 × 5:

- Có 4 ô vật cản cố định và 21 ô robot có thể đi qua.
- Robot bắt đầu tại `(0, 0)`.
- Mỗi episode mới, môi trường chọn ngẫu nhiên 3 ô bụi khác nhau trong 20 ô không phải vật cản hoặc ô xuất phát. Do đó số cách chọn bộ bụi ban đầu là `C(20, 3) = 1.140`.
- Robot có 4 hành động: lên, xuống, trái, phải.
- Khi robot đi vào ô có bụi, bụi được hút ngay trong bước đó; không có hành động hút riêng.
- Episode kết thúc thành công khi đã hút hết bụi, hoặc bị cắt khi đạt giới hạn 100 bước.

### Trạng thái

Trạng thái có dạng:

```text
s = (robot_x, robot_y, dust_bitmask)
```

`dust_bitmask` mã hóa các ô còn bụi thành một số nguyên: bit tương ứng với một ô bằng 1 nếu ô còn bụi, và bằng 0 nếu ô đã sạch. Cách biểu diễn này cho phép bảng Q nhận diện robot đang ở đâu và còn những vị trí bụi nào.

Có 21 vị trí hợp lệ cho robot. Vì bụi ban đầu được đặt trên 20 ô và tối đa có 3 hạt bụi, số mặt nạ bụi có thể có không vượt quá:

```text
C(20, 0) + C(20, 1) + C(20, 2) + C(20, 3) = 1.351
```

Do đó `21 × 1.351 = 28.371` là **cận trên lý thuyết** cho số trạng thái. Một số tổ hợp không thể xuất hiện trong thực tế, nên bảng Q không nhất thiết chứa đủ số trạng thái này.

### Phần thưởng

| Tình huống trong môi trường | Phần thưởng |
|---|---:|
| Di chuyển hợp lệ | `-1` |
| Va vào tường hoặc vật cản | `-20`, robot đứng yên |
| Đi vào ô có bụi và hút bụi | cộng `+20` |
| Hút sạch toàn bộ bụi | cộng thêm `+100` |

Phần thưởng được cộng dồn nếu nhiều điều kiện xảy ra trong cùng bước. Ví dụ, đi vào một ô bụi cuối cùng sẽ nhận `-1 + 20 + 100 = 119`.

## 3. Bảng Q và công thức cập nhật

`Q(s, a)` là ước lượng tổng phần thưởng chiết khấu mà robot có thể nhận được khi thực hiện hành động `a` tại trạng thái `s`, sau đó tiếp tục hành động tốt.

Sau khi thực hiện hành động, môi trường trả về trạng thái tiếp theo `s'`, phần thưởng `r`, và các cờ kết thúc. Giá trị Q được cập nhật theo:

```text
Q(s, a) ← Q(s, a) + α [target - Q(s, a)]

target = r                                  nếu episode kết thúc thành công
target = r + γ maxₐ' Q(s', a')              nếu chưa terminated
```

Trong đó:

- `α` (alpha) là tốc độ học: mức độ cập nhật Q theo thông tin mới.
- `γ` (gamma) là hệ số chiết khấu: mức độ coi trọng phần thưởng tương lai.
- `maxₐ' Q(s', a')` là giá trị lớn nhất trong các hành động có thể chọn ở trạng thái tiếp theo.

Nếu episode bị **truncated** do chạm giới hạn bước, mã vẫn bootstrap từ `s'`; chỉ trạng thái `terminated` do đã dọn sạch mới dùng mục tiêu `target = r`. Điều này tương ứng với việc giới hạn bước là điều kiện dừng quá trình mô phỏng, không phải trạng thái hoàn thành nhiệm vụ.

Sai số TD tại một bước là:

```text
TD error = target - Q(s, a)
```

## 4. Khám phá và khai thác

Trong lúc huấn luyện, chương trình dùng chính sách ε-greedy:

- Với xác suất `ε`, chọn ngẫu nhiên một trong bốn hành động để khám phá.
- Với xác suất `1 - ε`, chọn hành động có Q lớn nhất để khai thác kinh nghiệm đã học.
- Nếu nhiều hành động đồng hạng Q cao nhất, chương trình chọn ngẫu nhiên giữa các hành động đó.

Sau mỗi episode, epsilon giảm theo:

```text
ε ← max(ε_min, ε × epsilon_decay)
```

Khi trình diễn chính sách, chương trình không dùng bước khám phá ε nữa; nó chọn ngẫu nhiên chỉ giữa các hành động đồng hạng có Q cao nhất. Vì vậy robot có thể đi vòng hoặc quay lại nếu giá trị Q của các hướng ngang nhau hoặc chính sách chưa được học đủ tốt.

## 5. Các tham số trong chương trình

Giá trị mặc định trong `main()` khớp với cấu hình báo cáo:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| `episodes` | `600` | Số episode huấn luyện |
| `alpha` | `0.1` | Tốc độ học |
| `gamma` | `0.9` | Hệ số chiết khấu |
| `epsilon` | `1.0` | Xác suất khám phá ban đầu |
| `epsilon_decay` | `0.995` | Hệ số suy giảm epsilon sau mỗi episode |
| `min_epsilon` | `0.05` | Mức khám phá tối thiểu |
| `max_steps` | `100` | Giới hạn bước mỗi episode, lấy từ môi trường |

Q-Table được khởi tạo theo kiểu **lazy**: trạng thái nào được gặp thì mới thêm vào bảng với bốn giá trị Q ban đầu bằng 0. Cách này tương đương khởi tạo các giá trị Q bằng 0 nhưng tránh cấp phát trước toàn bộ bảng trạng thái.

## 6. Quy trình huấn luyện và chạy robot

1. Tạo môi trường và Q-Table rỗng.
2. Với mỗi episode huấn luyện:
   1. Sinh một bố trí bụi ngẫu nhiên mới, đồng thời giữ nguyên kích thước lưới, vật cản, vị trí xuất phát và giới hạn bước.
   2. Đặt lại robot về trạng thái ban đầu.
   3. Chọn hành động theo ε-greedy.
   4. Gọi `env.step(action)` để robot di chuyển; môi trường tự xử lý va chạm, hút bụi, phần thưởng và điều kiện kết thúc.
   5. Cập nhật Q theo công thức Q-Learning.
   6. Dừng episode khi `terminated` hoặc `truncated`, lưu tổng phần thưởng và giảm epsilon.
3. Sau huấn luyện, đặt lại môi trường để có một bố trí bụi cho lượt kiểm thử.
4. Chạy chính sách greedy đã học và hiển thị từng bước trên cửa sổ Matplotlib. Có thể chọn bản đồ ASCII thay cho giao diện đồ họa.
5. In kết quả gồm trạng thái thành công/thất bại, số bước, số lần va chạm và tổng phần thưởng.

### Mã giả

```text
Q ← bảng rỗng
ε ← 1.0

Lặp episodes lần:
    Tạo episode với 3 vị trí bụi ngẫu nhiên
    s ← env.reset()
    total_reward ← 0

    Trong khi episode chưa terminated hoặc truncated:
        Chọn a theo ε-greedy từ Q(s, ·)
        s', r, terminated, truncated ← env.step(a)

        Nếu terminated:
            target ← r
        Ngược lại:
            target ← r + γ × maxₐ' Q(s', a')

        Q(s, a) ← Q(s, a) + α × [target - Q(s, a)]
        s ← s'
        total_reward ← total_reward + r

    Lưu total_reward
    ε ← max(ε_min, ε × epsilon_decay)

Chạy một episode kiểm thử bằng hành động argmaxₐ Q(s, a)
```

## 7. Chạy chương trình

Trong Terminal tại thư mục dự án:

```powershell
python q_learning.py
```

Mặc định chương trình huấn luyện 600 episode rồi mở giao diện đồ họa cho một lượt chạy. Có thể thay đổi số episode và tốc độ hoạt ảnh:

```powershell
python q_learning.py --episodes 15000 --interval-ms 800
```

Để chạy bản đồ chữ trong terminal:

```powershell
python q_learning.py --render ascii
```

`--seed` cho phép tái lập chuỗi chọn bụi của quá trình huấn luyện và các lựa chọn ngẫu nhiên của agent:

```powershell
python q_learning.py --episodes 600 --seed 42
```

## 8. Giới hạn và lưu ý khi đánh giá

- 600 episode là tham số thực nghiệm theo cấu hình báo cáo, không đảm bảo agent đã học tốt trên mọi bố trí bụi. Có thể tăng số episode và đánh giá trên nhiều bố trí kiểm thử độc lập.
- Reward trung bình trong lúc huấn luyện được tính trên các episode có khám phá; không nên dùng riêng chỉ số này để kết luận chính sách greedy đã tốt. Nên báo cáo thêm tỷ lệ dọn sạch thành công, số bước trung bình và số va chạm trên nhiều episode kiểm thử.
- Q-Table học trên một cấu hình lưới và phần thưởng cụ thể; thay đổi vật cản, reward hoặc cách đặt bụi có thể cần huấn luyện lại.
- Môi trường là mô phỏng. Việc robot chạy trên cửa sổ đồ họa không điều khiển phần cứng máy hút bụi thật.
