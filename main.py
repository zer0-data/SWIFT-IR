import argparse
import cv2
import torch
import numpy as np
import time
from pathlib import Path

# Import custom modules
from src.classifier import WeatherClassifier
from src.preprocessing import FogEnhancer, LSRB
from src.detector import get_student_model

# Constants
CLASS_NAMES = ['Clear', 'Fog', 'Rain']
COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Green, Cyan, Blue

def parse_opt():
    parser = argparse.ArgumentParser(description="RAW-D: Robust Adaptive Weather Distillation Pipeline")
    parser.add_argument('--source', type=str, required=True, help='Path to video file or 0 for webcam')
    parser.add_argument('--weights_classifier', type=str, default='weights/classifier.pt', help='Path to Stage A weights')
    parser.add_argument('--weights_detector', type=str, default='weights/student_final.pt', help='Path to Stage C weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Detector confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='Display results in window')
    parser.add_argument('--save-vid', action='store_true', help='Save output video to runs/detect')
    return parser.parse_args()

class RawDPipeline:
    def __init__(self, opt):
        self.device = torch.device('cuda' if torch.cuda.is_available() and opt.device != 'cpu' else 'cpu')
        print(f"🚀 Initializing RAW-D Pipeline on {self.device}...")

        # --- Stage A: Classifier ---
        self.classifier = WeatherClassifier().to(self.device)
        self.classifier.load_state_dict(torch.load(opt.weights_classifier, map_location=self.device))
        self.classifier.eval()
        
        # --- Stage B: Weather Modules ---
        self.fog_module = FogEnhancer().to(self.device)
        self.rain_module = LSRB().to(self.device)
        # Note: Stage B modules often don't have learnable weights, or are loaded here if they do.
        
        # --- Stage C: Detector (Student) ---
        # We use the custom function to load the YOLO with ViT Neck
        self.detector = get_student_model(weights_path=opt.weights_detector, device=self.device)
        self.conf_thres = opt.conf

    def preprocess(self, img_bgr):
        """
        Normalize standard CV2 image (8-bit) to the 14-bit-like float format 
        expected by the network [0, 1].
        """
        # Resize to network input size (e.g., 640x640) could happen here or inside detector
        img = cv2.resize(img_bgr, (640, 640))
        img = img.astype(np.float32) / 255.0  # Normalize to 0-1
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Force 1-channel thermal
        
        # Convert to Tensor (B, C, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor, img_bgr

    def run(self, source, view_img=False, save_vid=False):
        # Open Video
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print(f"Error opening source: {source}")
            return

        # Video Writer Setup
        vid_writer = None
        if save_vid:
            save_path = Path('runs/detect') / f"output_{Path(source).name}"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            vid_writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        print("🔥 Starting Inference Loop...")
        prev_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Preprocess
            input_tensor, original_frame = self.preprocess(frame)

            # 2. Stage A: Classification Gate
            with torch.no_grad():
                class_logits = self.classifier(input_tensor)
                weather_id = torch.argmax(class_logits, dim=1).item()
                weather_name = CLASS_NAMES[weather_id]

            # 3. Dynamic Routing & Stage B
            processed_tensor = input_tensor
            
            if weather_id == 0:   # Clear
                pass # Zero Latency
            elif weather_id == 1: # Fog
                processed_tensor = self.fog_module(input_tensor)
            elif weather_id == 2: # Rain
                processed_tensor = self.rain_module(input_tensor)

            # 4. Stage C: Detector
            # YOLO expects list of results
            results = self.detector(processed_tensor, conf=self.conf_thres, verbose=False)

            # 5. Visualization & Post-processing
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Draw annotations (YOLOv8 plotting)
            # We plot on the original frame for display
            annotated_frame = results[0].plot(img=original_frame) 

            # Draw Custom Dashboard
            cv2.rectangle(annotated_frame, (10, 10), (250, 120), (0, 0, 0), -1) # BG Box
            cv2.putText(annotated_frame, f"Mode: {weather_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[weather_id], 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if weather_id != 0:
                cv2.putText(annotated_frame, "Module Active", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Output
            if view_img:
                cv2.imshow('RAW-D Pipeline', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if save_vid:
                vid_writer.write(annotated_frame)

        cap.release()
        if vid_writer:
            vid_writer.release()
            print(f"Saved result to {save_path}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = parse_opt()
    pipeline = RawDPipeline(opt)
    pipeline.run(opt.source, opt.view_img, opt.save_vid)