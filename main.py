# main.py
import argparse
from src import train, evaluate, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pneumonia Detection using CNN")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], default="train")
    parser.add_argument("--image", help="Path to X-ray image for prediction", default=None)
    args = parser.parse_args()

    if args.mode == "train":
        train.run_training()
    elif args.mode == "eval":
        evaluate.run_evaluation()
    elif args.mode == "predict":
        if not args.image:
            print("Please provide --image path for prediction mode")
        else:
            predict.run_prediction(args.image)

            