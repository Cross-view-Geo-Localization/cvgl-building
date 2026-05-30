# Hướng dẫn Training và Evaluation trên tập dữ liệu VisLoc

Toàn bộ các thiết lập quan trọng đều được quản lý thông qua file cấu hình .yaml.

## 1. Training
**Bước 1:** Tùy chỉnh Hyperparameters
Mở file config/visloc_sinkhorn.yaml để tinh chỉnh các tham số cấu hình cần thiết trước khi chạy (ví dụ: learning rate, batch size, số epochs, v.v.).

**Bước 2:** Huấn luyện tiếp từ Checkpoint
Nếu muốn tiếp tục quá trình training (resume) hoặc fine-tune từ một trọng số đã có, hãy mở file config/visloc_sinkhorn.yaml, tìm đến block training và điền đường dẫn weight vào biến checkpoint_start:

```YAML
# config/visloc_sinkhorn.yaml
training:
  checkpoint_start: "path/to/your/weight.pth" # Paste đường dẫn tới file trọng số tại đây
(Lưu ý: Để null nếu muốn train lại từ đầu).
```

**Bước 3:** Khởi chạy quá trình Training
Tại thư mục gốc của dự án, mở terminal và chạy lệnh sau:

```Bash
python train_semi_pos.py
```

# 2. Evaluation
**Bước 1:** Trỏ đúng file cấu hình
Trước khi chạy đánh giá, hệ thống cần biết cấu trúc model bạn đang sử dụng. Hãy mở script eval_scripts/eval_visloc.py và cập nhật lại đường dẫn file config cho khớp với model vừa train.

Ví dụ, chuyển từ config mặc định sang config của Sinkhorn:

```Python
# Bên trong file eval_scripts/eval_visloc.py
config = OmegaConf.load("./DroneCVGL/config/base.yaml") # Thay đổi từ base.yaml
```
**Bước 2:** Khởi chạy Evaluation
Tiến hành đánh giá bằng cách chạy lệnh sau trên terminal:

```Bash
python eval_scripts/eval_visloc.py
```