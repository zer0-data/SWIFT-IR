
"""Distillation training utilities for Teacher-Student YOLO training.

Provides:
- `DistillationLoss` : computes Ultralytics detection loss + MSE mimic loss.
- `DistillationTrainer` : simple PyTorch training loop that accepts pairs
  (raw_ir, pseudo_rgb), uses the frozen Teacher on `pseudo_rgb` to produce
  pseudo-labels/features, and trains the Student on `raw_ir` with combined loss.

Notes / design choices:
- This module makes a best-effort integration with the `ultralytics` YOLO
  wrapper. The student and teacher may be Ultralytics `YOLO` objects or raw
  `nn.Module` models. Where possible we call the internal loss routines; when
  unavailable the code raises informative errors.
- Feature extraction for the mimic loss is implemented with forward-hooks;
  we try to capture intermediate feature maps (preferably neck outputs). If
  multiple candidate maps exist we compute the mean MSE across matched maps.
"""
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F


class _HookCollector:
	"""Utility to collect forward activations for a module using hooks.

	Usage:
		collector = _HookCollector(model, filter_fn=...)  # installs hooks
		with collector.enabled():
			_ = model(input)
		activations = collector.activations  # list of (name, tensor)
	"""

	def __init__(self, model: nn.Module, filter_fn: Optional[callable] = None):
		self.model = model
		self.filter_fn = filter_fn or (lambda n, m: True)
		self._handles: List[torch.utils.hooks.RemovableHandle] = []
		self.activations: List[Tuple[str, torch.Tensor]] = []
		# register hooks on modules that match filter
		for name, mod in model.named_modules():
			try:
				if self.filter_fn(name, mod):
					h = mod.register_forward_hook(self._make_hook(name))
					self._handles.append(h)
			except Exception:
				# best-effort: skip modules that can't register hooks
				continue

	def _make_hook(self, name: str):
		def hook(_module, _input, output):
			# store a detached copy to avoid keeping computation graph
			try:
				# only store tensor-like activations
				if isinstance(output, torch.Tensor):
					self.activations.append((name, output.detach()))
				elif isinstance(output, (list, tuple)):
					# store the first tensor-like element if present
					for o in output:
						if isinstance(o, torch.Tensor):
							self.activations.append((name, o.detach()))
							break
			except Exception:
				pass

		return hook

	@contextmanager
	def enabled(self):
		try:
			# clear previous activations
			self.activations.clear()
			yield
		finally:
			# no-op; hooks remain installed until explicit remove
			pass

	def remove(self):
		for h in self._handles:
			try:
				h.remove()
			except Exception:
				pass
		self._handles.clear()


class DistillationLoss(nn.Module):
	"""Compute combined detection + mimic distillation loss.

	Args:
		alpha: weight multiplier for mimic (MSE) loss.
		reduction: reduction for MSE ('mean' recommended).
	"""

	def __init__(self, alpha: float = 1.0, reduction: str = "mean") -> None:
		super().__init__()
		self.alpha = float(alpha)
		self.mse = nn.MSELoss(reduction=reduction)

	def forward(
		self,
		yolo_loss: torch.Tensor,
		student_feats: List[torch.Tensor],
		teacher_feats: List[torch.Tensor],
	) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
		"""Combine losses.

		- `yolo_loss` is expected to be a scalar tensor (from Ultralytics or
		  another detector loss).
		- `student_feats` and `teacher_feats` are lists of tensors collected
		  from forward hooks. We compute MSE between matching shapes and
		  average the mimic losses.
		Returns (total_loss, info_dict)
		"""
		if not torch.is_tensor(yolo_loss):
			yolo_loss = torch.tensor(float(yolo_loss), device=student_feats[0].device if student_feats else None)

		mimic_loss = torch.tensor(0.0, device=yolo_loss.device)
		valid = 0
		# naive matching: for each student feature try to find a teacher feature
		# with same spatial dimensions and compute MSE. This keeps the API
		# simple and robust across model variants.
		for s in student_feats:
			for t in teacher_feats:
				if s.shape == t.shape:
					mimic_loss = mimic_loss + self.mse(s, t)
					valid += 1
					break

		if valid > 0:
			mimic_loss = mimic_loss / float(valid)
		else:
			# fallback: if no shapes matched, try to resize (bilinear) teacher
			# features to match the first student feature (best-effort).
			if student_feats and teacher_feats:
				s0 = student_feats[0]
				t0 = teacher_feats[0]
				try:
					t0r = F.interpolate(t0, size=s0.shape[-2:], mode="bilinear", align_corners=False)
					if t0r.shape[1] != s0.shape[1]:
						# if channel mismatch attempt a 1x1 conv-like projection
						minc = min(t0r.shape[1], s0.shape[1])
						mimic_loss = self.mse(s0[:, :minc], t0r[:, :minc])
					else:
						mimic_loss = self.mse(s0, t0r)
				except Exception:
					mimic_loss = torch.tensor(0.0, device=yolo_loss.device)

		total = yolo_loss + self.alpha * mimic_loss
		info = {"yolo_loss": yolo_loss.detach(), "mimic_loss": mimic_loss.detach(), "alpha": self.alpha}
		return total, info


class DistillationTrainer:
	"""Simple distillation trainer for Teacher-Student training.

	Behavior:
	  - Teacher is frozen and evaluated on `pseudo_rgb` to produce pseudo-labels
		and teacher features.
	  - Student is trained on `raw_ir` and we compute: total = yolo_loss + alpha*mimic.
	  - The trainer uses forward-hooks to capture intermediate features from
		both models. By default the hook filter prefers modules with 'neck'
		in their name (typical YOLO architecture), falling back to many
		activations if needed.

	NOTE: This implementation attempts to interoperate with `ultralytics.YOLO`.
	It expects the teacher/student to be either the YOLO wrapper (with
	`.model` attribute) or a plain `nn.Module`. When using the Ultralytics
	wrapper we attempt to use the model's internal loss computation by calling
	`model.model.compute_loss(preds, targets)` or `model.model._loss(...)`.
	If these are not present the code will raise an error describing the
	missing integration point.
	"""

	def __init__(
		self,
		student: Any,
		teacher: Any,
		optimizer: torch.optim.Optimizer,
		device: str = "cpu",
		alpha: float = 1.0,
	) -> None:
		self.device = torch.device(device)
		self.student = student
		self.teacher = teacher
		self.optimizer = optimizer
		self.alpha = float(alpha)

		# put models on device
		try:
			self.student.to(self.device)
		except Exception:
			pass
		try:
			self.teacher.to(self.device)
		except Exception:
			pass

		# freeze teacher
		self.teacher.eval()
		for p in self.teacher.parameters():
			p.requires_grad = False

		# collectors for hooks: prefer modules named with 'neck'
		def neck_filter(name, mod):
			low = name.lower()
			if "neck" in low or "c2f" in low or "backbone" in low:
				return True
			# also include conv layers close to the end if they appear
			if "m" == name.split(".")[-1] and hasattr(mod, "conv"):
				return True
			return False

		self.student_collector = _HookCollector(self._unwrap(self.student), filter_fn=neck_filter)
		self.teacher_collector = _HookCollector(self._unwrap(self.teacher), filter_fn=neck_filter)

		self.criterion = DistillationLoss(alpha=self.alpha)

	def _unwrap(self, model: Any) -> nn.Module:
		"""Return underlying nn.Module for Ultralytics `YOLO` wrappers or pass-through."""
		if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
			return model.model
		if isinstance(model, nn.Module):
			return model
		raise TypeError("Unsupported model type: expected nn.Module or YOLO wrapper with `.model`")

	def _yolo_loss_from_student(self, imgs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""Attempt to compute student detection loss using Ultralytics internals.

		The function tries common internal names and raises a helpful error if
		the expected API is not found.
		"""
		student_model = self._unwrap(self.student)

		# run student forward to get predictions
		preds = student_model(imgs)

		# try several common loss entrypoints
		# 1) compute_loss(preds, targets)
		if hasattr(student_model, "compute_loss"):
			out = student_model.compute_loss(preds, targets)
			# some implementations return (loss, loss_items)
			if isinstance(out, tuple):
				return out[0]
			return out

		# 2) _loss(preds, targets)
		if hasattr(student_model, "_loss"):
			out = student_model._loss(preds, targets)
			if isinstance(out, tuple):
				return out[0]
			return out

		# 3) try model.loss(preds, targets)
		if hasattr(student_model, "loss"):
			try:
				out = student_model.loss(preds, targets)
				if isinstance(out, tuple):
					return out[0]
				return out
			except Exception:
				pass

		raise RuntimeError(
			"Unable to compute YOLO loss: the student model does not expose a supported loss API."
			" If using Ultralytics, ensure the underlying `.model` has `compute_loss` or `_loss`."
		)

	@staticmethod
	def _results_to_targets(results: Any, imgs_shape: List[Tuple[int, int]]) -> torch.Tensor:
		"""Convert Ultralytics inference Results into training targets tensor.

		Expected target format: (N,6) with columns [img_idx, class, x, y, w, h]
		where x,y,w,h are normalized to [0,1] and x,y are center coordinates.

		`results` is expected to be the ultralytics output object iterable per-image,
		where each item exposes `.boxes.xyxy` and `.boxes.cls` or similar.
		This function is best-effort and will skip images with no detections.
		"""
		rows: List[torch.Tensor] = []
		for i, r in enumerate(results):
			# try to support both ultralytics v8 Results and simpler lists
			try:
				boxes = getattr(r, "boxes", None)
				if boxes is None:
					# maybe r is a plain ndarray Nx6 (xyxy, conf, cls)
					arr = getattr(r, "boxes_xyxy", None) or r
					arr = torch.as_tensor(arr)
					if arr.numel() == 0:
						continue
					# fallback parsing not implemented fully
					continue

				# ultralytics Boxes: .xyxy (tensor Nx4) and .cls (tensor Nx1)
				xyxy = boxes.xyxy.cpu()  # (N,4)
				cls = boxes.cls.cpu() if hasattr(boxes, "cls") else torch.zeros((xyxy.shape[0],), dtype=torch.long)
				h_img, w_img = imgs_shape[i]
				if xyxy.numel() == 0:
					continue
				# convert to xywh normalized
				x1y1 = xyxy[:, :2]
				x2y2 = xyxy[:, 2:4]
				centers = (x1y1 + x2y2) / 2.0
				wh = (x2y2 - x1y1)
				# normalize
				centers[:, 0] = centers[:, 0] / float(w_img)
				centers[:, 1] = centers[:, 1] / float(h_img)
				wh[:, 0] = wh[:, 0] / float(w_img)
				wh[:, 1] = wh[:, 1] / float(h_img)

				for j in range(centers.shape[0]):
					c = torch.tensor([
						float(i),
						float(cls[j].item() if cls.numel() else 0.0),
						float(centers[j, 0].item()),
						float(centers[j, 1].item()),
						float(wh[j, 0].item()),
						float(wh[j, 1].item()),
					])
					rows.append(c)
			except Exception:
				continue

		if not rows:
			# no targets -> return empty tensor shape (0,6)
			return torch.zeros((0, 6), dtype=torch.float32)
		return torch.stack(rows, dim=0)

	def train_step(self, raw_ir: torch.Tensor, pseudo_rgb: torch.Tensor) -> Dict[str, Any]:
		"""Run a single training step.

		Args:
			raw_ir: Tensor batch of raw thermal images (B,1,H,W) or (B,H,W).
			pseudo_rgb: Tensor batch of pseudo RGB inputs to the teacher (B,3,H,W) or compatible.

		Returns:
			info dict with losses and metrics.
		"""
		# ensure batches on device
		raw_ir = raw_ir.to(self.device)
		pseudo_rgb = pseudo_rgb.to(self.device)

		# 1) Run teacher -> get pseudo-labels and teacher features
		teacher_mod = self._unwrap(self.teacher)
		student_mod = self._unwrap(self.student)

		with torch.no_grad():
			with self.teacher_collector.enabled():
				teacher_results = self.teacher(pseudo_rgb) if hasattr(self.teacher, "__call__") else teacher_mod(pseudo_rgb)
				teacher_feats = [t for _, t in self.teacher_collector.activations]

		# prepare image shapes for target normalization
		imgs_shape = []
		# accept both (B,C,H,W) and (B,H,W)
		if pseudo_rgb.dim() == 4:
			for b in range(pseudo_rgb.shape[0]):
				h = int(pseudo_rgb.shape[2])
				w = int(pseudo_rgb.shape[3])
				imgs_shape.append((h, w))
		else:
			imgs_shape = [(raw_ir.shape[-2], raw_ir.shape[-1])] * raw_ir.shape[0]

		pseudo_targets = self._results_to_targets(teacher_results, imgs_shape)

		# 2) Run student forward to gather features and compute predictions
		with self.student_collector.enabled():
			# student forward should return predictions compatible with loss fn
			preds = self.student(raw_ir) if hasattr(self.student, "__call__") else student_mod(raw_ir)
			student_feats = [t for _, t in self.student_collector.activations]

		# 3) Compute yolo loss using student internals and pseudo_targets
		# move targets to device
		if pseudo_targets.numel() > 0:
			pseudo_targets = pseudo_targets.to(self.device)

		try:
			yolo_loss = self._yolo_loss_from_student(raw_ir, pseudo_targets)
		except Exception as e:
			raise RuntimeError(f"Failed to compute YOLO loss: {e}")

		# 4) combine losses
		total_loss, info = self.criterion(yolo_loss, student_feats, teacher_feats)

		# 5) backward + step
		self.optimizer.zero_grad()
		total_loss.backward()
		self.optimizer.step()

		# return losses as scalars
		info_out = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in info.items()}
		info_out.update({"total_loss": float(total_loss.detach().cpu().item())})
		return info_out


__all__ = ["DistillationLoss", "DistillationTrainer"]

