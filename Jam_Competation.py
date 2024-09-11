import cv2
import numpy as np

# Define frame dimensions and aspect ratio
frame_width = 640
frame_height = 480
aspect_ratio = frame_width / frame_height

# Capture video from webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define color thresholds for ball detection (adjust as needed)
lower_color = np.array([20, 100, 100])  # Lower HSV values for tennis ball color
upper_color = np.array([30, 255, 255])  # Upper HSV values for tennis ball color

# Create virtual screen
virtual_screen = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

# Calibration flag
calibrated = False

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask for tennis ball color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours of detected objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect ball hit
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Filter out small contours (noise)
        if area > 100:
            # Find center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw circle at impact point on virtual screen
                cv2.circle(virtual_screen, (cX, cY), 10, (0, 0, 0), -1)

    # Calibration (press 'c' to calibrate)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        # Prompt user to ensure the frame is visible and the ball is in the center
        print("Calibration starting. Ensure the frame is visible and place the ball in the center.")
        cv2.waitKey(3000)  # Wait for 3 seconds to allow adjustment

        # Capture frame for calibration
        ret, calibration_frame = cap.read()
        if ret:
            # Detect the ball in the calibration frame
            calibration_hsv = cv2.cvtColor(calibration_frame, cv2.COLOR_BGR2HSV)
            calibration_mask = cv2.inRange(calibration_hsv, lower_color, upper_color)
            calibration_contours, _ = cv2.findContours(calibration_mask, cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)

            if calibration_contours:
                # Assume the largest contour is the ball
                largest_contour = max(calibration_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Calculate scaling factors based on the detected ball position
                # Assuming the ball should be in the center of the frame
                horizontal_scaling = frame_width / w
                vertical_scaling = frame_height / h

                # Set calibrated flag
                calibrated = True
                print("Calibration successful.")
            else:
                print("Calibration failed. Ball not detected.")
        else:
            print("Calibration failed. Could not capture frame.")

    # Apply scaling if calibrated
    if calibrated:
        # Resize virtual screen to match frame dimensions while maintaining aspect ratio
        virtual_screen_resized = cv2.resize(virtual_screen, (frame_width, frame_height))
        cv2.imshow("Virtual Screen", virtual_screen_resized)
    else:
        cv2.imshow("Virtual Screen", virtual_screen)

    # Display video feed
    cv2.imshow("Video Feed", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()