import argparse
import cv2
import torch
import numpy as np
import time
from pathlib import Path

from src.pipeline import SwiftIRPipeline

# Display helpers
CLASS_NAMES = ["Clear", "Fog", "Rain"]
MODE_COLORS = {"Clear": (0, 200, 0), "Fog": (0, 200, 200), "Rain": (200, 0, 0)}


def parse_opt():
    parser = argparse.ArgumentParser(description="SWIFT-IR: Adaptive Weather-Aware Detection Pipeline")
    parser.add_argument('--source', type=str, default='0', help='Path to video file or 0 for webcam')
    parser.add_argument('--weights_classifier', type=str, default='weights/classifier.pt', help='Path to classifier weights')
    parser.add_argument('--weights_detector', type=str, default='weights/student_final.pt', help='Path to detector weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Detector confidence threshold')
    parser.add_argument('--device', type=str, default='cpu', help='Device string, e.g. cpu or cuda')
    parser.add_argument('--save-vid', action='store_true', help='Save output video to runs/detect')
    parser.add_argument('--view-img', action='store_true', help='Display results in window')
    return parser.parse_args()


def prepare_frame_for_pipeline(frame: np.ndarray, target_size=(640, 640)) -> torch.Tensor:
    # Convert BGR -> Gray, resize, normalize to [0,1], return torch.Tensor (H,W) float32
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    arr = resized.astype(np.float32) / 255.0
    return torch.from_numpy(arr).to(dtype=torch.float32)


def main():
    opt = parse_opt()

    src = opt.source
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        print(f"Error opening source: {src}")
        return

    # Initialize pipeline
    pipeline = SwiftIRPipeline(
        classifier_weights=opt.weights_classifier,
        yolo_weights=opt.weights_detector,
        device=opt.device,
    )

    # Video writer
    vid_writer = None
    if opt.save_vid:
        out_dir = Path("runs/detect")
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"output_{Path(src).stem}.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        vid_writer = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    prev = time.time()

    window_name = "SWIFT-IR"
    if opt.view_img:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Starting inference loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare input tensor for pipeline
        inp = prepare_frame_for_pipeline(frame)  # (H,W) float32

        # Run pipeline
        try:
            results = pipeline.forward(inp)
        except Exception as e:
            print(f"Pipeline error: {e}")
            results = None

        # Determine mode from classifier by running classifier predict directly for UI
        try:
            with torch.no_grad():
                cls_id = int(pipeline.classifier.predict_image(inp, device=pipeline.device))
        except Exception:
            cls_id = 0

        mode_name = CLASS_NAMES[cls_id]
        mode_color = MODE_COLORS.get(mode_name, (255, 255, 255))

        # Visualization
        annotated = frame.copy()
        if results is not None:
            try:
                # ultralytics Results-like object
                annotated = results[0].plot(img=annotated)
            except Exception:
                pass

        # Dashboard box
        cv2.rectangle(annotated, (10, 10), (280, 110), (0, 0, 0), -1)
        cv2.putText(annotated, f"Mode: {mode_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
        if mode_name != "Clear":
            cv2.putText(annotated, "Module Active", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev) if (now - prev) > 0 else 0.0
        prev = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if opt.view_img:
            cv2.imshow(window_name, annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if vid_writer is not None:
            vid_writer.write(annotated)

    cap.release()
    if vid_writer is not None:
        vid_writer.release()
        print(f"Saved video to {fname}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()