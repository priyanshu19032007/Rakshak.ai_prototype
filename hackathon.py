import cv2
import numpy as np

cap = cv2.VideoCapture(0)

_, prev_frame = cap.read()

print("System Ready...")
print("1. Yellow Object = FIRE")
print("2. Blue Object = WEAPON (Knife/Gun)")
print("3. Movement = SUSPICIOUS MOTION")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- 1. FIRE LOGIC (Yellow/Orange) ---
    lower_fire = np.array([22, 120, 120], dtype="uint8") 
    upper_fire = np.array([35, 255, 255], dtype="uint8")
    fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
    fire_count = cv2.countNonZero(fire_mask)

    # --- 2. WEAPON LOGIC (Blue Object) ---
    # Hum maan ke chal rahe hain Blue color = Weapon hai (Demo ke liye)
    lower_weapon = np.array([100, 150, 0], dtype="uint8") 
    upper_weapon = np.array([140, 255, 255], dtype="uint8")
    weapon_mask = cv2.inRange(hsv, lower_weapon, upper_weapon)
    weapon_count = cv2.countNonZero(weapon_mask)

    status_text = "STATUS: SAFE"
    status_color = (0, 255, 0) # Green

    # PRIORITY 1: FIRE (Sabse dangerous)
    if fire_count > 500:
        status_text = "ALERT: FIRE DETECTED! (101)"
        status_color = (0, 0, 255) # Red
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
    
    # PRIORITY 2: WEAPON (Blue Object)
    elif weapon_count > 1000: # Sensitivity adjust kar sakte ho
        status_text = "ALERT: WEAPON DETECTED! (Knife/Gun)"
        status_color = (128, 0, 128) # Purple Color for Weapon
        
        # Weapon ke charo taraf box banao
        contours, _ = cv2.findContours(weapon_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 0, 128), 3)
                cv2.putText(frame, "Weapon", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

    # PRIORITY 3: MOTION (Agar Fire aur Weapon nahi hai tab check karo)
    else:
        diff = cv2.absdiff(prev_frame, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_contour = None
        max_area = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000: # Sirf badi movement
                if area > max_area:
                    max_area = area
                    largest_contour = contour

        if largest_contour is not None:
            status_text = "SUSPICIOUS MOTION"
            status_color = (0, 165, 255) # Orange
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display Text
    cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow("Smart City - Final Demo", frame)

    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()