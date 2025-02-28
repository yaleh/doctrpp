import argparse

import cv2
import paddle

from .GeoTr import GeoTr
from .utils import to_image, to_tensor


def predict(image_path, model_path, output_path):
    """
    Predict using the GeoTr model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the model checkpoint.
        output_path (str): Path to save the output image.
    """
    checkpoint = paddle.load(model_path)
    state_dict = checkpoint["model"]
    model = GeoTr()
    model.set_state_dict(state_dict)
    model.eval()

    img_org = cv2.imread(image_path)
    img = cv2.resize(img_org, (288, 288))
    x = to_tensor(img)
    y = to_tensor(img_org)
    bm = model(x)
    bm = paddle.nn.functional.interpolate(
        bm, y.shape[2:], mode="bilinear", align_corners=False
    )
    bm_nhwc = bm.transpose([0, 2, 3, 1])
    out = paddle.nn.functional.grid_sample(y, (bm_nhwc / 288 - 0.5) * 2)
    out_image = to_image(out)
    cv2.imwrite(output_path, out_image)
    
    print(f"Processed image saved to {output_path}")


def main():
    """Command line interface for doctrpp."""
    parser = argparse.ArgumentParser(description="DocTrPP - Document Image Processing with PaddlePaddle")

    parser.add_argument(
        "--image", "-i", 
        required=True,
        type=str, 
        help="Path to the input image"
    )

    parser.add_argument(
        "--model", "-m", 
        required=True,
        type=str, 
        help="Path to the model checkpoint"
    )

    parser.add_argument(
        "--output", "-o", 
        required=True,
        type=str, 
        help="Path to save the output image"
    )

    args = parser.parse_args()
    
    print(f"Processing image: {args.image}")
    print(f"Using model: {args.model}")
    print(f"Output will be saved to: {args.output}")
    
    predict(args.image, args.model, args.output)


if __name__ == "__main__":
    main() 