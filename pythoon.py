import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI (box where hand should be)
    roi = frame[100:400, 100:400]

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Flip for mirror view (optional but useful)
    frame = cv2.flip(frame, 1)
    # Define ROI (box where hand should be)
    roi = frame[100:400, 100:400]

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)    

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define skin color range
    lower_skin = np.array([0, 30, 60])
    upper_skin = np.array([20, 150, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Define kernel
    kernel = np.ones((3, 3), np.uint8)

    # Remove noise
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
    # Find largest contour (assume it's hand)
        max_contour = max(contours, key=cv2.contourArea)

    # Draw contour
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
    # Convex hull
        hull = cv2.convexHull(max_contour)

# Draw hull
    cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

# Convexity defects
    hull_indices = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull_indices)

    finger_count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]

            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])

        # Draw points
            cv2.circle(frame, far, 5, (255, 0, 0), -1)

        # Count fingers
            if d > 20000:
                finger_count += 1

# Show count
    cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    

    # Show outputs
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()