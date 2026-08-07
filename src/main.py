import cv2

# Create a connection with the default webcam

cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Continuously read frames
while True:

    # Read one frame
    ret, frame = cap.read()
    #Check if the frame was captured
    if not ret:
        print("Error: Could not read frame.")
        break


    # Display the frame
    cv2.imshow("AirType AI", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()