from collections import deque
import math
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

#保存每个目标的历史轨迹
track_history={}

track_last_seen = {}

track_time_history = {}

#平滑速度记录
speed_ema = {}
alpha = 0.2
# 4. 实时检测循环
while True:

    # 读取摄像头画面
    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头画面")
        break

    # YOLO + ByteTrack 目标跟踪
    results = model.track(
        source=frame,
        device=device,
        imgsz=640,
        conf=0.25,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # 记录当前检测帧的时间
    frame_time = time.perf_counter()

    # 获取检测结果
    boxes = results[0].boxes

    # 第一帧打印检测信息
    if frame_count == 0:
        print("检测目标数量：", len(boxes))

        for box in boxes:
            print("类别：", model.names[int(box.cls[0].item())])
            print("置信度：", box.conf[0].item())
            print("跟踪ID：", box.id[0].item() if box.id is not None else "无")
            print("-" * 30)

    frame_count += 1

    # 5. 准备绘制画面
    annotated_frame = frame.copy()

    # 每一帧重新统计
    class_counts = {}
    total_count = len(boxes)

    # 6. 遍历所有检测目标
    for box in boxes:

        # 获取坐标
        x1, y1, x2, y2 = (
            box.xyxy[0].cpu().numpy().astype(int)
        )

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

        # 记录目标运动轨迹
        if track_id != -1:

            # 计算中心点
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            center = (center_x, center_y)

            # 首次出现的目标，初始化两个队列
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=30)
                track_time_history[track_id] = deque(maxlen=30)

            # 保存坐标
            track_history[track_id].append(center)

            # 保存对应时间
            track_time_history[track_id].append(frame_time)
            # 获取历史轨迹
            points = track_history[track_id]

            # 获取对应的时间历史
            times = track_time_history[track_id]

            # 默认速度
            speed = 0.0

            # 初始化默认值
            direction = "UNKNOWN"

            smooth_speed=0.0

            # 至少有两个坐标才能判断方向
            if len(points) >= 2:

                previous = points[-2]
                current = points[-1]

                dx = current[0] - previous[0]
                dy = current[1] - previous[1]

                # 计算时间间隔
                elapsed_time = times[-1] - times[-2]

                # 计算运动距离
                distance = math.hypot(dx, dy)

                # 计算像素速度
                if elapsed_time > 0:

                    # 计算当前速度
                    speed = distance / elapsed_time

                    # 第一次测量，直接使用当前速度初始化
                    if track_id not in speed_ema:
                        speed_ema[track_id] = speed

                    else:
                        # EMA 指数移动平均
                        speed_ema[track_id] = (
                                alpha * speed
                                + (1 - alpha) * speed_ema[track_id]
                        )

                    # 获取平滑后的速度
                    smooth_speed = speed_ema[track_id]

                threshold = 5

                if abs(dx) < threshold and abs(dy) < threshold:
                    direction = "STOP"

                elif abs(dx) >= abs(dy):
                    direction = "RIGHT" if dx > 0 else "LEFT"

                else:
                    direction = "DOWN" if dy > 0 else "UP"

            track_last_seen[track_id] = frame_count

        # 统计类别数量
        class_counts[class_name] = (
            class_counts.get(class_name, 0) + 1
        )

        # 绘制检测框
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # 生成标签
        if track_id != -1:
            label = (
                f"ID:{track_id} {class_name}"
                f" {confidence:.2f}{direction}"
                f"{smooth_speed:.1f}px/s"
            )
        else:
            label = f"{class_name} {confidence:.2f}"

        # 绘制类别、ID和置信度
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 找出过期目标
    expired_ids = []

    for track_id, last_seen in track_last_seen.items():

        if frame_count - last_seen > 30:
            expired_ids.append(track_id)

    # 删除过期目标
    for track_id in expired_ids:
        track_history.pop(track_id, None)

        track_last_seen.pop(track_id, None)

        track_time_history.pop(track_id, None)

        speed_ema.pop(track_id, None)
    # 绘制目标运动轨迹
    for track_id, points in track_history.items():

        for i in range(1, len(points)):
            cv2.line(
                annotated_frame,
                points[i - 1],
                points[i],
                (255, 0, 0),
                2
            )

    # 7. 绘制目标总数
    cv2.putText(
        annotated_frame,
        f"Objects: {total_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # 8. 绘制类别统计
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

    # 9. FPS统计：每处理一帧，只增加一次
    fps_frame_count += 1

    current_time = time.perf_counter()
    elapsed_time = current_time - fps_start_time

    # 每隔0.5秒更新一次FPS
    if elapsed_time >= 0.5:

        fps = fps_frame_count / elapsed_time

        fps_frame_count = 0
        fps_start_time = current_time

    # 显示FPS
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    # 10. 显示画面
    cv2.imshow("YOLO + OpenCV", annotated_frame)

    # 按Q退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 11. 释放资源
cap.release()
cv2.destroyAllWindows()

print("程序已退出")
