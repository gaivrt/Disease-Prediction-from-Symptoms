"""Command-line interface for disease prediction."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import train_model, train_all_models
from evaluate import evaluate_model, evaluate_all_models
from models import MODEL_NAMES


def main():
    parser = argparse.ArgumentParser(description='Disease Prediction CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', choices=MODEL_NAMES + ['all'], default='all',
                              help='Model to train (default: all)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model', choices=MODEL_NAMES + ['all'], default='all',
                             help='Model to evaluate (default: all)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        if args.model == 'all':
            train_all_models()
        else:
            result = train_model(args.model)
            print(f"\n{result['model_name']} trained successfully!")
            print(f"  Accuracy: {result['validation_accuracy']:.4f}")
            print(f"  Macro F1: {result['validation_f1']:.4f}")
            print(f"  Saved to: {result['save_path']}")
            
    elif args.command == 'evaluate':
        if args.model == 'all':
            evaluate_all_models()
        else:
            result = evaluate_model(args.model)
            print(f"\n{result['model_name']} Evaluation:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Macro F1: {result['f1_macro']:.4f}")
            print(f"\nClassification Report:\n{result['classification_report']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
