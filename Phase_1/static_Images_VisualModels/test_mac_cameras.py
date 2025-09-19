import cv2

for idx in range(5):  # test first 5 indexes
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        print(f"Camera index {idx} is available ✅")
        cap.release()
    else:
        print(f"Camera index {idx} not available ❌")
