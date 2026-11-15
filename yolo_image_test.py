from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo26n.pt")

print("YOLO模型加载成功！")