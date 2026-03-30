import argparse
import torch
import numpy as np
from imageio import imread, imsave
from matplotlib import cm
from Other.utility_functions import PSNR, ssim


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR and SSIM between two images.")
    parser.add_argument("image", help="Path to the test image")
    parser.add_argument("gt", help="Path to the ground truth image")
    parser.add_argument("-o", "--output", default="loss.png", help="Path for the loss image (default: loss.png)")
    args = parser.parse_args()

    img = torch.tensor(imread(args.image), dtype=torch.float32) / 255
    gt = torch.tensor(imread(args.gt), dtype=torch.float32) / 255

    p = PSNR(img, gt)
    s = ssim(img.permute(2, 0, 1).unsqueeze(0), gt.permute(2, 0, 1).unsqueeze(0))

    print(f"PSNR: {p:0.03f} dB")
    print(f"SSIM: {s:0.03f}")

    error = torch.mean((img - gt) ** 2, dim=2).clamp(0, 1).numpy()
    loss_img = (cm.turbo(4 * error)[:, :, :3] * 255).astype(np.uint8)
    imsave(args.output, loss_img)
    print(f"Loss image saved to {args.output}")


if __name__ == "__main__":
    main()