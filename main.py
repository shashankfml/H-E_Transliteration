#!/usr/bin/env python3
"""
Main script for Seq2Seq transliteration model training and evaluation.
"""

import torch
import torch.nn as nn
import argparse
from data_preparation import DataPreparation
from models import Encoder, Decoder, Seq2Seq
from training_utils import train_loop, evaluate_with_beam_search
from sweep_config import run_sweep


def main():
    parser = argparse.ArgumentParser(description='Seq2Seq Transliteration Model')
    parser.add_argument('--mode', choices=['train', 'sweep', 'evaluate'], default='train',
                       help='Mode: train single model, run sweep, or evaluate with beam search')
    parser.add_argument('--datapath', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--encoder_embedding_dim', type=int, default=256,
                       help='Encoder embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    parser.add_argument('--cell_type', choices=['LSTM', 'GRU', 'RNN'], default='LSTM',
                       help='RNN cell type')
    parser.add_argument('--attention', action='store_true',
                       help='Use attention mechanism')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', choices=['Adam', 'NAdam', 'RAdam', 'AdamW'], default='Adam',
                       help='Optimizer')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                       help='Teacher forcing ratio')
    parser.add_argument('--beam_width', type=int, default=1,
                       help='Beam width for decoding')
    parser.add_argument('--weight_decay', type=float, default=0,
                       help='Weight decay')
    parser.add_argument('--save_path', type=str, default='.',
                       help='Path to save model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for evaluation')

    args = parser.parse_args()

    if args.mode == 'sweep':
        run_sweep()
        return

    # Load dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_func = DataPreparation(args.datapath)
    train_dataloader, validation_dataloader, test_dataloader = dataset_func.create_dataloaders(args.batch_size)

    if args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: --model_path required for evaluation mode")
            return
        
        # Load saved model
        model = torch.load(args.model_path, map_location=device)
        model.eval()
        
        # Evaluate with beam search
        print(f"Evaluating with beam search (beam_width={args.beam_width})...")
        accuracy = evaluate_with_beam_search(model, test_dataloader, args.beam_width, device)
        print(f"Beam Search Accuracy: {accuracy:.4f}")
        return

    # Single model training
    dataset_func = DataPreparation(args.datapath)
    train_dataloader, validation_dataloader, test_dataloader = dataset_func.create_dataloaders(args.batch_size)

    # Create model
    enc = Encoder(dataset_func.num_encoder_tokens, args.hidden_dim, args.n_layers, 
                 args.dropout, args.encoder_embedding_dim, args.cell_type)
    dec = Decoder(dataset_func.num_decoder_tokens, args.hidden_dim, args.n_layers, 
                 args.dropout, args.encoder_embedding_dim, args.cell_type, atten=args.attention)
    model = Seq2Seq(enc, dec, dataset_func.max_source_length, dataset_func.max_target_length,
                   dataset_func.target_char2int, dataset_func.num_decoder_tokens, device)
    model = model.to(device)

    # Create config object
    class Config:
        def __init__(self, args):
            for key, value in vars(args).items():
                setattr(self, key, value)

    config = Config(args)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_func.target_char2int["-PAD-"])
    
    # Train model
    train_loop(model, train_dataloader, validation_dataloader, device, criterion, 
              config=config, save_path=args.save_path, clip=1, sweep=False)


if __name__ == "__main__":
    main()
