import argparse
from doctrpp.cli import predict

def main():
    parser = argparse.ArgumentParser(description="DocTrPP Predict from Package")

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
    
    print(f"Using doctrpp package to process image: {args.image}")
    predict(args.image, args.model, args.output)

if __name__ == "__main__":
    main() 