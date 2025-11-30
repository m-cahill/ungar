import sys
import torch

def main():
    if torch.cuda.is_available():
        print("ERROR: CUDA is available but we expected CPU-only torch.")
        print(f"Torch version: {torch.__version__}")
        sys.exit(1)
    
    print(f"SUCCESS: Torch {torch.__version__} is CPU-only as expected.")

if __name__ == "__main__":
    main()

