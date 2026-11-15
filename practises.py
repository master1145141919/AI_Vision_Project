import cv2
import torch
import time
from pathlib import Path
from ultralytics import YOLO

# 1. 加载模型
weights_path = Path(__file__).parent / "weights" / "yolo26n.pt"

model = YOLO(str(weights_path))

# 2. 选择计算设备
device = 0 if torch.cuda.is_available() else "cpu"

print("当前设备：", device)

# 3. 打开摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")

frame_count = 0

# FPS统计变量
fps = 0
fps_frame_count = 0
fps_start_time = time.perf_counter()

# 4. 实时检测循环
while True:

    #111start_time = time.perf_counter()

    # 读取摄像头画面
    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头画面")
        break

    # YOLO 进行目标检测
    results = model.track(
        source=frame,
        device=device,
        imgsz=640,
        conf=0.25,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    # 提取检测结果
    boxes = results[0].boxes

    # 只打印第一帧，避免控制台刷屏
    if frame_count == 0:
        print("检测目标数量：", len(boxes))

        for i in range(len(boxes)):
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 获取置信度
            confidence = box.conf[0].item()

            # 获取类别编号
            class_id = int(box.cls[0].item())

            # 获取类别名称
            class_name = model.names[class_id]

            print("类别：", class_name)
            print("置信度：", round(confidence, 3))
            print("坐标：", x1, y1, x2, y2)
            print("-" * 30)

    frame_count += 1

    # 绘制检测结果
    # 复制原始图像
    annotated_frame = frame.copy()

    class_counts = {}

    # 遍历所有检测框
    for box in results[0].boxes:
        # 获取坐标
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # 获取置信度
        confidence = box.conf[0].item()

        # 获取类别编号
        class_id = int(box.cls[0].item())

        # 获取类别名称
        class_name = model.names[class_id]

        # 获取跟踪ID
        if box.id is not None:
            track_id = int(box.id[0].item())
        else:
            track_id = -1

        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 绘制矩形框
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 统计目标总数
        total_count = len(results[0].boxes)

        # 显示目标总数
        cv2.putText(
            annotated_frame,
            f"Objects: {total_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # 显示每种类别的数量
        for i, (name, count) in enumerate(class_counts.items()):
            cv2.putText(
                annotated_frame,
                f"{name}: {count}",
                (20, 105 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # 计算 FPS
        # FPS统计
        fps_frame_count += 1

        current_time = time.perf_counter()
        elapsed_time = current_time - fps_start_time

        # 每隔0.5秒更新一次FPS
        if elapsed_time >= 0.5:
            fps = fps_frame_count / elapsed_time

            fps_frame_count = 0
            fps_start_time = current_time

        # 显示 FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # 生成标签
        if track_id != -1:
            label = f"ID:{track_id} {class_name} {confidence:.2f}"
        else:
            label = f"{class_name} {confidence:.2f}"

        # 绘制类别和置信度
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        cv2.imshow("YOLO + OpenCV", annotated_frame)
    # 显示画面
    cv2.imshow("YOLO + OpenCV", annotated_frame)

    # 按 Q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 5. 释放资源
cap.release()
cv2.destroyAllWindows()

print("程序已退出")




def get_center(box):
    x1, y1, x2, y2 = box
    center_x = int(x1 + x2) / 2
    center_y = int(y1 + y2) / 2
    return center_x,center_y