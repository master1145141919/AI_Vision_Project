import time

import cv2

# 1. 打开电脑默认摄像头
cap = cv2.VideoCapture(0)

# 2. 检查摄像头是否打开
if not cap.isOpened():
    print("摄像头打开失败！")
    exit()

print("摄像头启动成功！")
print("按 Q 键退出")

start_time = time.time()
# 3. 循环读取画面
while True:
    ret, frame = cap.read()

    if not ret:
        print("读取画面失败！")
        break
    end_time = time.time()

    fps = 1 / (end_time - start_time)

    start_time = end_time

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 4. 显示画面
    cv2.imshow("AI Vision Camera", frame)

    # 5. 按 Q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 释放资源
cap.release()
cv2.destroyAllWindows()

print("摄像头已关闭")
