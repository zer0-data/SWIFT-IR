
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
	import yaml
except Exception:  # pragma: no cover - yaml optional at import time
	yaml = None


def extract_stats_features(images: torch.Tensor) -> torch.Tensor:
	"""Downsample a batch of 1-channel thermal images to 64x64 and extract
	four statistical features per image: variance, max, min, mean.

	Args:
		images: Tensor of shape (B, 1, H, W) or (B, H, W).

	Returns:
		Tensor of shape (B, 4) with features in order [variance, max, min, mean].
	"""
	if not isinstance(images, torch.Tensor):
		raise TypeError("images must be a torch.Tensor")

	# Normalize shape to (B,1,H,W)
	if images.ndim == 3:
		images = images.unsqueeze(1)
	elif images.ndim != 4:
		raise ValueError("images must have 3 or 4 dims (B,H,W) or (B,1,H,W)")

	# Ensure float
	imgs = images.float()

	# Downsample to 64x64 for speed using adaptive average pooling (fast)
	imgs_ds = F.adaptive_avg_pool2d(imgs, output_size=(64, 64))

	# Flatten spatial dims
	B = imgs_ds.shape[0]
	flat = imgs_ds.view(B, -1)

	# Compute statistics
	mean = flat.mean(dim=1)
	var = flat.var(dim=1, unbiased=False)
	mn = flat.min(dim=1).values
	mx = flat.max(dim=1).values

	features = torch.stack([var, mx, mn, mean], dim=1)
	return features


class WeatherClassifier(nn.Module):
	"""A small MLP classifier for weather conditions.

	Architecture: Input(4) -> Linear(hidden1) -> BN -> ReLU -> Linear(hidden2)
	-> BN -> ReLU -> Linear(num_classes)
	"""

	def __init__(
		self,
		hidden1: int = 64,
		hidden2: int = 32,
		num_classes: int = 3,
		class_names: typing.Optional[typing.List[str]] = None,
	) -> None:
		super().__init__()
		self.fc1 = nn.Linear(4, hidden1)
		self.bn1 = nn.BatchNorm1d(hidden1)
		self.fc2 = nn.Linear(hidden1, hidden2)
		self.bn2 = nn.BatchNorm1d(hidden2)
		self.fc3 = nn.Linear(hidden2, num_classes)

		if class_names is None:
			self.class_names = ["Clear", "Fog", "Rain"]
		else:
			self.class_names = class_names

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward expects input tensor of shape (B, 4) features and returns logits."""
		x = self.fc1(x)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = self.fc3(x)
		return x

	def predict_image(self, image: torch.Tensor, device: typing.Optional[str] = None) -> int:
		"""Run inference on a single raw image tensor and return integer class id.

		Args:
			image: Tensor with shape (H,W), (1,H,W) or (B,1,H,W) where B=1.
			device: Optional device string like 'cpu' or 'cuda'. If provided, model
					and tensors will be moved there for inference.

		Returns:
			int: class id (0=Clear, 1=Fog, 2=Rain)
		"""
		self.eval()
		if device is None:
			device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")
		if isinstance(device, str):
			device = torch.device(device)

		# Prepare image -> (1,1,H,W)
		img = image
		if img.ndim == 2:
			img = img.unsqueeze(0).unsqueeze(0)
		elif img.ndim == 3:
			# could be (1,H,W) or (B,1,H,W)
			if img.shape[0] == 1:
				img = img.unsqueeze(0)  # (1,1,H,W)
			else:
				# assume (B,1,H,W)
				pass
		elif img.ndim == 4:
			pass
		else:
			raise ValueError("Unsupported image tensor shape for predict_image")

		img = img.to(device=device, dtype=torch.float32)

		with torch.no_grad():
			feats = extract_stats_features(img)
			feats = feats.to(device=device)
			logits = self.forward(feats)
			pred = int(logits.argmax(dim=1).item())

		return pred


def load_config(path: typing.Union[str, Path]):
	"""Load YAML config from path. Returns dict or raises if yaml unavailable."""
	if yaml is None:
		raise RuntimeError("PyYAML not available; install pyyaml to use load_config")
	p = Path(path)
	with p.open("r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f)
	return cfg


__all__ = [
	"extract_stats_features",
	"WeatherClassifier",
	"load_config",
]

