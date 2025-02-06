import cv2
import numpy as np
import time

class BasicVisionTracking:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize color tracking
        self.lower_color = np.array([35, 50, 50])  # Green color in HSV
        self.upper_color = np.array([85, 255, 255])
        
        # Motion tracking variables
        self.prev_frame = None
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    def detect_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small detections
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Green Object', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return
        
        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small movements
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Motion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        self.prev_frame = gray

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count >= 30:
            end_time = time.time()
            self.fps = self.frame_count / (end_time - self.start_time)
            self.frame_count = 0
            self.start_time = time.time()

    def start(self):
        cap = cv2.VideoCapture(0)  # Use default camera
        
        print("Starting camera feed... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Run detections
            self.detect_faces(frame)
            self.detect_color(frame)
            self.detect_motion(frame)
            
            # Calculate and display FPS
            self.calculate_fps()
            cv2.putText(frame, f'FPS: {int(self.fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show the frame
            cv2.imshow('Vision Tracking', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BasicVisionTracking()
    tracker.start()
