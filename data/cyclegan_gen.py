

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms


class ResnetBlock(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.conv_block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
			nn.InstanceNorm2d(dim),
			nn.ReLU(True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
			nn.InstanceNorm2d(dim),
		)

	def forward(self, x):
		return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
	def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
		assert n_blocks >= 0
		super().__init__()
		model = [nn.ReflectionPad2d(3),
				 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
				 nn.InstanceNorm2d(ngf),
				 nn.ReLU(True)]

		# downsample
		n_downsampling = 2
		mult = 1
		for i in range(n_downsampling):
			mult_prev = mult
			mult = mult * 2
			model += [
				nn.Conv2d(ngf * mult_prev, ngf * mult, kernel_size=3, stride=2, padding=1, bias=False),
				nn.InstanceNorm2d(ngf * mult),
				nn.ReLU(True)
			]

		# resnet blocks
		mult = 2 ** n_downsampling
		for i in range(n_blocks):
			model += [ResnetBlock(ngf * mult)]

		# upsample
		for i in range(n_downsampling):
			mult_prev = mult
			mult = mult // 2
			model += [
				nn.ConvTranspose2d(ngf * mult_prev, ngf * mult, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
				nn.InstanceNorm2d(ngf * mult),
				nn.ReLU(True)
			]

		model += [nn.ReflectionPad2d(3),
				  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
				  nn.Tanh()]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		return self.model(input)


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


class ImageFolderDataset(Dataset):
	def __init__(self, root: str, transform=None, recursive=True):
		self.root = Path(root)
		self.transform = transform
		files: List[Path] = []
		if recursive:
			for p in self.root.rglob('*'):
				if p.suffix.lower() in IMAGE_EXTS and p.is_file():
					files.append(p)
		else:
			for p in self.root.iterdir():
				if p.suffix.lower() in IMAGE_EXTS and p.is_file():
					files.append(p)

		files.sort()
		self.files = files

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		p = self.files[idx]
		img = Image.open(p).convert('RGB')
		if self.transform:
			img_t = self.transform(img)
		else:
			img_t = transforms.ToTensor()(img)
		return img_t, str(p)


def load_generator(weights_path: str, device: torch.device):
	model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
	model.to(device)
	ckpt = torch.load(weights_path, map_location=device)
	# support both raw state_dict and dict with 'state_dict'
	if isinstance(ckpt, dict) and 'state_dict' in ckpt:
		state = ckpt['state_dict']
	else:
		state = ckpt
	try:
		model.load_state_dict(state)
	except Exception:
		# try stripping module prefix
		new_state = {}
		for k, v in state.items():
			nk = k.replace('module.', '')
			new_state[nk] = v
		model.load_state_dict(new_state, strict=False)
	model.eval()
	return model


def tensor_to_pil(img_tensor: torch.Tensor):
	# img_tensor: (B, C, H, W) or (C, H, W) in range [-1,1]
	if img_tensor.dim() == 4:
		img_tensor = img_tensor[0]
	img = (img_tensor.clamp(-1, 1) + 1) / 2.0
	img = (img * 255.0).byte().cpu()
	np_img = img.permute(1, 2, 0).numpy()
	return Image.fromarray(np_img)


def save_batch(outputs: torch.Tensor, paths: List[str], input_root: Path, output_root: Path):
	outputs = outputs.detach()
	for i in range(outputs.size(0)):
		out_pil = tensor_to_pil(outputs[i])
		in_path = Path(paths[i])
		rel = in_path.relative_to(input_root)
		out_path = output_root.joinpath(rel)
		out_path.parent.mkdir(parents=True, exist_ok=True)
		out_pil.save(out_path)


def main():
	parser = argparse.ArgumentParser(description='Generate Pseudo-RGB images from thermal images using a CycleGAN generator')
	parser.add_argument('--input_dir', required=True, help='Folder with thermal images')
	parser.add_argument('--weights', required=True, help='Path to pretrained generator .pth')
	parser.add_argument('--output_dir', required=True, help='Folder to save pseudo-RGB images')
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'), help='Optional resize to W H')
	parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
	parser.add_argument('--num_workers', type=int, default=4)
	args = parser.parse_args()

	device = torch.device(args.device)

	transform_list = []
	if args.size:
		transform_list.append(transforms.Resize((args.size[1], args.size[0])))
	transform_list += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
	transform = transforms.Compose(transform_list)

	dataset = ImageFolderDataset(args.input_dir, transform=transform, recursive=True)
	if len(dataset) == 0:
		print(f'No images found in {args.input_dir}')
		return

	dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

	model = load_generator(args.weights, device)

	input_root = Path(args.input_dir).resolve()
	output_root = Path(args.output_dir).resolve()
	output_root.mkdir(parents=True, exist_ok=True)

	try:
		from tqdm import tqdm
		iterator = tqdm(dataloader, desc='Processing')
	except Exception:
		iterator = dataloader

	with torch.no_grad():
		for batch in iterator:
			imgs, paths = batch
			imgs = imgs.to(device)
			out = model(imgs)
			save_batch(out, paths, input_root, output_root)

	print(f'Done. Generated images saved to: {output_root}')


if __name__ == '__main__':
	main()

