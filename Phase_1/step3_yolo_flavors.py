# The Flavors of YOLOv8 with COCO (Common Object in Context)

# Dute maaa... the small model starts detecting lapotp for the first time 
# And mouse.. and Chair.. starts to have some confidence and starts biding elements together.
# Wow man... this is absolutely Fascinating.
# The Experience of the Development of the Mind in the Evolutionary "Open Your Eyes" String - as it begin to grasp more and more and Inference better and better.
# And plants.. and other things 
# Man as a General Model - it's fantastic! And the feeling man..
# The actual physical substrate that frames are being sent in continously and the Neural Net performs inference on and on and on.
# Absolutely fascinating.


# Medium Model: Wow man.. it starts discerning more and more.
# It sees the comfy chair as both chair and couch - which is actually is - a blend of chair and couch.
# Man - just as the Language Models - beginning to grasp the Pixelated world and how it forms concepts and concepts go deeper than pixels (letters) - is extraordinary!


# Large Model: hmmm.. even though FPS are not dropping - the prediction quality is similar to the Medium model - 
# BUT it's starting to pick up much less objects - this is interesting and weird - like it's being more selective - 
# and only picking up higher inferences or not even trying to grasp others -  as a performance in the environment of my room - i'd rate it lower than medium

# Ah but be mindful that we are also dealing with teh limited classifications of COCO - so if the object is not in COCO, it won't be detected.
# This is absolute key aspect as we move forward - into beginning to prepare for training of the Model for our needs -

# Yet this raises the questions - what are other techniques and models beyond YOLOV8 - that can be used for Object Detection ?
# And what are some limitations towards what we'd consider Visual Model *AGI* - HLP - that is to pick up a full range of objects - and being able to discern EVERYTHING 
# (it'd also mean it processes the image in a way - but then if something becomes relevant in some Scoring dimension like importance, threat, gain, etc - 
# it will then focus on that part of the SAME IMAGE (the same matrix of pixels) - but inference is now performed only at 5-10% of the image - going deeper nad deeper and discerning more and more.


# And for the first time - the Mac has gotten Warm.
# The Neural Nets computing inside locally.
# What a Feeling.


from ultralytics import YOLO
import cv2
import warnings
warnings.filterwarnings("ignore")


# Load YOLOv8 nano (fastest)
model = YOLO("yolov8m.pt")
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
