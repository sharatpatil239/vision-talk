import os
import time
import uuid
import cv2

# Define target labels and capture count
LABELS = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
NUMBER_IMGS = 15
IMAGES_PATH = os.path.join('workspace', 'images', 'collectedimages')

os.makedirs(IMAGES_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
for label in LABELS:
    label_path = os.path.join(IMAGES_PATH, label)
    os.makedirs(label_path, exist_ok=True)
    
    print(f"\n==========================================")
    print(f"GET READY: Collecting images for '{label}' in 5 seconds...")
    print(f"==========================================")
    time.sleep(5)
    
    for img_num in range(NUMBER_IMGS):
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        img_name = os.path.join(label_path, f"{label}.{str(uuid.uuid1())}.jpg")
        cv2.imwrite(img_name, frame)
        cv2.imshow('Image Collection Feed', frame)
        print(f"Captured ({img_num + 1}/{NUMBER_IMGS}) for {label}")
        
        # 2-second pause to change hand angle/orientation
        time.sleep(2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nImage collection complete!")