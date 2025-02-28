#!/usr/bin/env python3
"""
Simple test script for doctrpp package.
"""

import sys
from doctrpp.cli import predict

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input_image> <model_path> <output_image>")
        sys.exit(1)
        
    input_image = sys.argv[1]
    model_path = sys.argv[2]
    output_image = sys.argv[3]
    
    print(f"Processing image: {input_image}")
    print(f"Using model: {model_path}")
    print(f"Output will be saved to: {output_image}")
    
    predict(input_image, model_path, output_image)
    print("Done!")

if __name__ == "__main__":
    main() 