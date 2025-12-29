"""Main pipeline stitching classifier, weather modules, and YOLO detector for inference.

Class: SwiftIRPipeline

Init: Load the Classifier, FogModule, RainModule, and YOLO Detector from their weights.

Preprocess: Load 14-bit raw TIFF images and normalize to 0-1 float32.

Forward: Hard routing logic based on classifier prediction:
    - Clear: pass input to YOLO
    - Fog: pass input -> FogModule -> YOLO
    - Rain: pass input -> RainModule -> YOLO

Returns YOLO detection results (bounding boxes and metadata).
"""
from typing import Optional, Tuple, Any
from pathlib import Path

import torch
import numpy as np

from .classifier import WeatherClassifier, extract_stats_features
from .preprocessing import FogEnhancer, LSRB
from .detector import get_student_model


def _read_tiff(path: Path) -> np.ndarray:
    """Read TIFF file returning numpy array (H,W) with original integer depth.

    Tries tifffile, then imageio, then PIL as fallback.
    """
    try:
        import tifffile

        arr = tifffile.imread(str(path))
        return arr
    except Exception:
        pass

    try:
        import imageio

        arr = imageio.imread(str(path))
        return arr
    except Exception:
        pass

    try:
        from PIL import Image

        img = Image.open(str(path))
        arr = np.asarray(img)
        return arr
    except Exception as e:
        raise RuntimeError(f"Failed to read TIFF {path}: {e}")


class SwiftIRPipeline:
    """Pipeline that routes images through restoration modules before detection.

    Args:
        classifier_weights: Optional path to classifier checkpoint (torch .pt). If None,
            a fresh `WeatherClassifier` instance is used.
        fog_weights: Optional path (currently unused; FogEnhancer has no weights).
        rain_weights: Optional path (currently unused; LSRB has no weights).
        yolo_config: YOLO config path for the student model (defaults to configs/yolov8_student.yaml).
        yolo_weights: Optional weights path for YOLO detector.
        device: device string like 'cpu' or 'cuda'.
    """

    def __init__(
        self,
        classifier_weights: Optional[str] = None,
        fog_weights: Optional[str] = None,
        rain_weights: Optional[str] = None,
        yolo_config: str = "configs/yolov8_student.yaml",
        yolo_weights: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        # Classifier
        self.classifier = WeatherClassifier()
        if classifier_weights:
            try:
                ckpt = torch.load(classifier_weights, map_location=self.device)
                if isinstance(ckpt, dict) and "model" in ckpt:
                    state = ckpt["model"]
                else:
                    state = ckpt
                self.classifier.load_state_dict(state, strict=False)
            except Exception:
                # Ignore load errors; keep fresh model
                pass
        self.classifier.to(self.device)
        self.classifier.eval()

        # Fog and Rain modules (stateless / small models)
        self.fog = FogEnhancer()
        self.rain = LSRB()
        # if there were weights for these, one could load them here; provided args accepted for API
        self.fog.to(self.device)
        self.rain.to(self.device)
        self.fog.eval()
        self.rain.eval()

        # YOLO student detector
        self.yolo = get_student_model(config_path=yolo_config, weights=yolo_weights, device=str(self.device))

    def preprocess(self, tiff_path: str) -> torch.Tensor:
        """Load a 14-bit TIFF and normalize to float32 in [0,1].

        Returns a torch.Tensor shaped (H, W) dtype float32 on CPU.
        """
        p = Path(tiff_path)
        if not p.exists():
            raise FileNotFoundError(f"TIFF not found: {p}")

        arr = _read_tiff(p)

        # Ensure 2D
        if arr.ndim == 3 and arr.shape[2] == 3:
            # single-channel expected; if RGB provided, convert to luminance by simple mean
            arr = arr.mean(axis=2)

        arr = np.asarray(arr)

        # Determine bit depth from dtype or max value; assume up to 16-bit, default target max for 14-bit is 16383
        max_possible = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else float(arr.max() or 1.0)
        # Prefer 14-bit scaling if values are within that range
        if arr.max() <= 16383 and max_possible >= 16383:
            scale = 16383.0
        else:
            scale = max_possible

        arr = arr.astype(np.float32) / float(scale)
        arr = np.clip(arr, 0.0, 1.0)

        return torch.from_numpy(arr).to(dtype=torch.float32)

    def _run_yolo(self, img: torch.Tensor) -> Any:
        """Run the YOLO model and return detection results.

        Accepts `img` as torch tensor (H,W) or (1,H,W) or (B,1,H,W).
        Returns the raw ultralytics results object (or model output).
        """
        # Normalize shapes to HWC numpy for ultralytics where possible
        # Convert (H,W) or (1,H,W) -> (H,W,1)
        t = img
        if isinstance(t, torch.Tensor):
            t_cpu = t.detach().cpu()
            if t_cpu.ndim == 2:
                arr = t_cpu.numpy()
                arr = np.expand_dims(arr, axis=-1)
            elif t_cpu.ndim == 3 and t_cpu.shape[0] == 1:
                arr = t_cpu.squeeze(0).numpy()
                arr = np.expand_dims(arr, axis=-1)
            elif t_cpu.ndim == 4:
                # (B,1,H,W) -> (B,H,W,1)
                arr = t_cpu.permute(0, 2, 3, 1).numpy()
            else:
                arr = t_cpu.numpy()
        else:
            arr = np.asarray(t)

        try:
            # ultralytics accepts list of arrays or single array
            res = self.yolo(arr)
        except Exception:
            # fallback: try predict
            try:
                res = self.yolo.predict(arr)
            except Exception as e:
                raise RuntimeError(f"YOLO inference failed: {e}")

        return res

    def forward(self, img: torch.Tensor) -> Any:
        """Run one image through classifier -> optional module -> YOLO.

        Args:
            img: torch.Tensor in shape (H,W) or (1,H,W) or (B,1,H,W) with values in [0,1].

        Returns:
            YOLO detection results (raw results from ultralytics).
        """
        # Ensure float32 and single sample
        t = img.to(dtype=torch.float32).to(self.device)

        # Classifier expects single-sample and will compute stats internally
        cls_id = int(self.classifier.predict_image(t, device=self.device))

        # Route through modules
        if cls_id == 0:
            routed = t
        elif cls_id == 1:
            # Fog
            with torch.no_grad():
                routed = self.fog(t.unsqueeze(0) if t.dim()==2 else t)
        elif cls_id == 2:
            # Rain
            with torch.no_grad():
                routed = self.rain(t.unsqueeze(0) if t.dim()==2 else t)
        else:
            routed = t

        # Ensure routed is CPU numpy array for YOLO call
        if isinstance(routed, torch.Tensor):
            routed_cpu = routed.detach().cpu()
            # if shape (1,1,H,W) -> (H,W)
            if routed_cpu.ndim == 4 and routed_cpu.shape[0] == 1 and routed_cpu.shape[1] == 1:
                routed_cpu = routed_cpu.squeeze(0).squeeze(0)
        else:
            routed_cpu = routed

        results = self._run_yolo(routed_cpu)
        return results


__all__ = ["SwiftIRPipeline"]

