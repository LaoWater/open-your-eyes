# Instead of FasterRCNN architecture and models such as resnet
# We use MobileNet-SSD or YOLOv5n/YOLOv8n — they’re way faster and designed for real-time detection.
# We just want to run and feel the Model in current mac archutecture - then we'll move onto next step - make relevant results for our interest less random: gloves, hands, people, cups, etc
# We will pursue and try to improve on b oth realms of HLP Inference - and quality of the application ~ Feel, flow, speed. We don't want laggy, clunky, slow, unsatisfying experience.

# Oh yes man - quality fucking video stream! A pleasure to watch
# Yet the Model feels very small - it is far, far, far, far away from Human Level Performance.
# But it's a great start. We are merely exploring
# Now naturally we'd want a stronger model - as the base ~Feel of this one is that 
# The neural net is not strong enough to capture and fulfill the task at our hands: Real-Time Resturant Safety detection.
# Let's continue exploring - and if hitting some hardware bottlenecks, we move onto PC & GPUs (or even Cloud)


# Important Notes: if slow inference, we could resize teh feeded frame to smaller size - thus making the needed compute() faster, as the input would be smaller.
# the n (nano) models are designed to trade IQ for FPS. They’re meant for running on microcontrollers, Raspberry Pis, or CPU-only laptops.
# In Real Inference, we'd use much bigger ones.
# Well unless if the context requires it - maybe in a simple scenario we can train this nano model and run the device on a small edge device. (different context than our deeper restaurant policy)


from ultralytics import YOLO
import cv2

# Load YOLOv8 nano (fastest)
model = YOLO("yolov8n.pt")
# model.to("mps") # bit laggy on this M2 ? feels better without.


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    # Annotate
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
