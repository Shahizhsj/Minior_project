import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("model.pt")

# Open the video file
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        try:
            annotated_frame = results[0].plot()  # Ensure the method is supported
        except AttributeError as e:
            print(f"Error visualizing results: {e}")
            break

        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Print an error and exit if the frame can't be read
        print("Failed to read a frame. Exiting...")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
