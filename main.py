import argparse
import sys
from scripts.prepare_data import create_manifest
from scripts.smoke_test import smoke_test
from scripts.train import train
from scripts.evaluate import evaluate
from scripts.predict import predict

def main():
    parser = argparse.ArgumentParser(description="SIH Oil Spill SAR Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparsers for commands
    parser_prep = subparsers.add_parser("prepare", help="Generate train/val/test split manifest")
    parser_smoke = subparsers.add_parser("smoke-test", help="Run a fast smoke test of the model architecture")
    parser_train = subparsers.add_parser("train", help="Train the model")
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate the model on the test set")
    parser_predict = subparsers.add_parser("predict", help="Run inference and generate GeoJSON polygons")

    args = parser.parse_args()

    if args.command == "prepare":
        create_manifest()
    elif args.command == "smoke-test":
        smoke_test()
    elif args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "predict":
        predict()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
