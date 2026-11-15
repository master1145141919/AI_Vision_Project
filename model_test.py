from pathlib import Path
import torch
from ultralytics import YOLO

# 1. 定位模型文件
weights_path = Path(__file__).resolve().parent / "weights" / "yolo26n.pt"

if not weights_path.exists():
    raise FileNotFoundError(f"模型文件不存在：{weights_path}")

# 2. 检查 GPU
print("=" * 40)
print("YOLO26 模型测试")
print("=" * 40)

print("CUDA 是否可用：", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU：", torch.cuda.get_device_name(0))
    device = 0
else:
    device = "cpu"

# 3. 加载本地 YOLO 模型
print("\n正在加载 YOLO26n 模型...")

model = YOLO(str(weights_path))

print("模型加载成功！")
print("模型任务：", model.task)

# 4. 使用摄像头进行实时目标检测
print("\n开始实时目标检测...")

results = model.predict(
    source=0,
    device=device,
    imgsz=640,
    conf=0.25,
    show=True,
    verbose=True
)

print("检测结束！")