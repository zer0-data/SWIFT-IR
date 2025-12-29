

import typing
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
	"""Lightweight transformer block operating on spatial features.

	Input: (B, C, H, W)
	Internally flattens spatial dims, applies LayerNorm -> MHA -> MLP, then reshapes back.
	"""

	def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
		super().__init__()
		self.norm1 = nn.LayerNorm(dim)
		self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
		self.norm2 = nn.LayerNorm(dim)
		hidden_dim = max(1, int(dim * mlp_ratio))
		self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.shape
		# (B, C, H, W) -> (B, N, C)
		x_flat = x.flatten(2).permute(0, 2, 1)
		x_norm = self.norm1(x_flat)
		attn_out, _ = self.attn(x_norm, x_norm, x_norm)
		x = x_flat + attn_out
		x = x + self.mlp(self.norm2(x))
		# (B, N, C) -> (B, C, H, W)
		x = x.permute(0, 2, 1).view(b, c, h, w)
		return x


class C2f_TRT(nn.Module):
	"""C2f-like block where the bottleneck is replaced by a tiny Transformer (TRT) block.

	Designed to be a drop-in replacement for a standard C2f when used in most YOLOv8 backbones.
	The module preserves input->output shape semantics.
	"""

	def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
		super().__init__()
		self.c1 = c1
		self.c2 = c2
		self.n = n
		hidden = max(1, c2 // 2)
		# reduce channels -> transformer -> expand channels
		self.reduce = nn.Conv2d(c1, hidden, kernel_size=1, stride=1, padding=0, bias=False)
		# small transformer on reduced channels
		num_heads = max(1, min(8, hidden // 16))
		self.trt = TransformerBlock(hidden, num_heads=num_heads)
		self.expand = nn.Conv2d(hidden, c2, kernel_size=1, stride=1, padding=0, bias=False)
		self.act = nn.Identity()
		self.shortcut = shortcut

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.reduce(x)
		y = self.trt(y)
		y = self.expand(y)
		if self.shortcut and x.shape == y.shape:
			return x + y
		return y


def _replace_c2f_with_trt(module: nn.Module) -> None:
	"""Recursively replace instances named 'C2f' with `C2f_TRT` where possible."""
	for name, child in list(module.named_children()):
		clsname = child.__class__.__name__
		if clsname == "C2f":
			# best-effort: try to extract typical attributes, else fall back to in/out channels
			c1 = getattr(child, 'c1', None)
			c2 = getattr(child, 'c2', None)
			n = getattr(child, 'n', 1)
			if c1 is None or c2 is None:
				# attempt to infer from first conv if present
				try:
					conv = next(child.children())
					c1 = getattr(conv, 'in_channels', c1)
					c2 = getattr(conv, 'out_channels', c2)
				except Exception:
					pass
			if c1 is None or c2 is None:
				# fallback: don't replace if we can't infer channels
				continue
			new = C2f_TRT(c1=c1, c2=c2, n=n)
			setattr(module, name, new)
		else:
			_replace_c2f_with_trt(child)


def get_student_model(config_path: str = "configs/yolov8_student.yaml", weights: typing.Optional[str] = None, device: str = "cpu"):
	"""Build the student YOLO model from `config_path`, replace C2f blocks with C2f_TRT,
	and optionally load `weights` if provided.

	- Requires the `ultralytics` package to build a YOLO model from a YAML config.
	- If `weights` is a checkpoint dict saved by `torch.save(...)`, the function will
	  attempt to load the `model` key or the dict directly (non-strict).
	"""
	try:
		from ultralytics import YOLO
	except Exception as e:
		raise ImportError("ultralytics package is required to build the YOLO model. Install via `pip install ultralytics`.") from e

	# Build model from the YAML config
	model = YOLO(config_path)

	# Replace C2f blocks with our Transformer-backed variant
	try:
		_replace_c2f_with_trt(model.model)
	except Exception:
		# best-effort; don't fail if replacement can't be performed
		pass

	# Optionally load weights (torch checkpoint or ultralytics .pt)
	if weights:
		import torch as _torch
		ckpt = _torch.load(weights, map_location=device)
		state = None
		if isinstance(ckpt, dict):
			state = ckpt.get('model', ckpt)
		else:
			state = ckpt
		try:
			model.model.load_state_dict(state, strict=False)
		except Exception:
			# fallback to ultralytics loader if available
			try:
				model.load(weights)
			except Exception as e:
				raise RuntimeError(f"Failed to load weights from {weights}: {e}")

	model.to(device)
	return model

