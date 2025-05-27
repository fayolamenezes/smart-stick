import torch
import cv2
import pyttsx3
import threading
import numpy as np

class ObjectDetector:
    def __init__(self, model_name='yolov5s', conf_threshold=0.3, crowd_threshold=5):
        print("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded successfully!")

        self.conf_threshold = conf_threshold
        self.crowd_threshold = crowd_threshold
        self.hazardous_objects = self.get_hazardous_objects()
        self.tts_engine = self.init_tts()

    def get_hazardous_objects(self):
        return {
            'tree', 'wall', 'fence', 'car', 'motorcycle', 'bicycle', 'bench',
            'garbage bin', 'scaffolding', 'truck', 'traffic light', 'gate',
            'table', 'chair', 'door', 'shelf', 'stairs', 'escalator',
            'rug', 'elevator', 'cable', 'dog', 'cat', 'cow', 'puddle'
        }

    def init_tts(self):
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 1.0)
        return tts_engine

    def text_to_speech(self, message):
        threading.Thread(target=self._speak, args=(message,)).start()

    def _speak(self, message):
        print(f"Speaking: {message}")
        self.tts_engine.say(message)
        self.tts_engine.runAndWait()

    def detect_objects(self, frame):
        results = self.model(frame)
        return results.pandas().xyxy[0]

    def process_frame(self, frame):
        if frame is None:
            print("Warning: Empty frame detected, skipping...")
            return None

        frame_resized = cv2.resize(frame, (640, 480))  # Resize for faster processing
        results = self.detect_objects(frame_resized)
        detected_hazards = set()
        person_count = 0

        for _, row in results.iterrows():
            if row['confidence'] < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            label = row['name']

            if label == 'person':
                person_count += 1

            if label in self.hazardous_objects:
                detected_hazards.add(label)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{label}: {row['confidence']:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if person_count >= self.crowd_threshold:
            detected_hazards.add("crowd")
            self.text_to_speech("Warning! Crowd detected ahead.")
        elif person_count == 1:
            self.text_to_speech("Warning! 1 person detected ahead.")
        elif person_count > 1:
            self.text_to_speech(f"Warning! {person_count} persons detected ahead.")

        for hazard in detected_hazards:
            self.text_to_speech(f"Warning! {hazard} detected ahead.")

        return frame

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_stairs_and_direction(frame):
    results = model(frame)
    df = results.pandas().xyxy[0]
    stair_detected = False

    for _, row in df.iterrows():
        if row['name'] == 'stairs' and row['confidence'] > 0.3:
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            stair_detected = True
            stair_region = frame[y1:y2, x1:x2]
            depth_map = get_depth_map(stair_region)

            avg_top = np.mean(depth_map[:depth_map.shape[0]//3])
            avg_bottom = np.mean(depth_map[-depth_map.shape[0]//3:])
            
            stair_direction = "downward" if avg_top > avg_bottom else "upward"
            return "Stairs detected, direction: " + stair_direction

    return "No stairs detected"

def get_depth_map(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        depth = midas(img)
    return depth.squeeze().numpy()

def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture("C:/Users/Dell Pc/Downloads/car.mp4")

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Processing video...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            processed_frame = detector.process_frame(frame)
            stair_msg = detect_stairs_and_direction(frame)
            print(stair_msg)

            if processed_frame is not None:
                cv2.imshow('Object & Depth Detection', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Detection stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
