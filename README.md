# Seq2Seq Transliteration Model - Refactored

This is a refactored version of the original Jupyter notebook implementation, organized into modular Python files for better maintainability and reusability.

## Project Structure

```
├── data_preparation.py    # Data loading and preprocessing
├── models.py             # Neural network models (Encoder, Decoder, Seq2Seq, Attention)
├── training_utils.py     # Training and evaluation utilities
├── sweep_config.py       # Weights & Biases sweep configuration
├── main.py              # Main training script
├── requirements.txt     # Python dependencies
└── README_refactored.md # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Model Training

Train a single model with custom parameters:

```bash
python main.py --mode train \
    --datapath /path/to/dataset \
    --epochs 30 \
    --batch_size 128 \
    --hidden_dim 512 \
    --attention \
    --cell_type LSTM \
    --learning_rate 0.001
```

### Hyperparameter Sweep

Run a Weights & Biases sweep for hyperparameter optimization:

```bash
python main.py --mode sweep
```

## Module Descriptions

### `data_preparation.py`
- `DataPreparation`: Handles data loading, preprocessing, and dataloader creation
- Supports train/validation/test splits
- Creates character-level vocabularies and one-hot encodings

### `models.py`
- `Attention`: Attention mechanism for the decoder
- `Encoder`: RNN-based encoder (LSTM/GRU/RNN)
- `Decoder`: RNN-based decoder with optional attention
- `Seq2Seq`: Complete sequence-to-sequence model

### `training_utils.py`
- `train()`: Training loop for one epoch
- `evaluate()`: Evaluation loop
- `train_loop()`: Complete training process
- `accuracy_calc()`: Accuracy calculation utility

### `sweep_config.py`
- Weights & Biases sweep configuration
- Hyperparameter search space definition
- Sweep execution functions

## Key Features

- **Modular Design**: Clean separation of concerns
- **Flexible Architecture**: Support for LSTM/GRU/RNN cells
- **Attention Mechanism**: Optional attention for improved performance
- **Hyperparameter Optimization**: Built-in W&B sweep support
- **GPU Support**: Automatic CUDA detection and usage

## Command Line Arguments

- `--mode`: Choose between 'train' (single model) or 'sweep' (hyperparameter search)
- `--datapath`: Path to dataset directory
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--hidden_dim`: Hidden dimension size
- `--n_layers`: Number of RNN layers
- `--encoder_embedding_dim`: Embedding dimension
- `--dropout`: Dropout rate
- `--cell_type`: RNN cell type (LSTM/GRU/RNN)
- `--attention`: Enable attention mechanism
- `--learning_rate`: Learning rate
- `--optimizer`: Optimizer choice (Adam/NAdam/RAdam/AdamW)
- `--teacher_forcing_ratio`: Teacher forcing ratio during training
- `--save_path`: Directory to save trained model

## Example Usage

```python
# Import modules for custom usage
from data_preparation import DataPreparation
from models import Encoder, Decoder, Seq2Seq
from training_utils import train_loop

# Load data
data_prep = DataPreparation('/path/to/data')
train_loader, val_loader, test_loader = data_prep.create_dataloaders(batch_size=128)

# Create model
encoder = Encoder(data_prep.num_encoder_tokens, hidden_dim=512, n_layers=2, dropout=0.4)
decoder = Decoder(data_prep.num_decoder_tokens, hidden_dim=512, n_layers=2, dropout=0.4, atten=True)
model = Seq2Seq(encoder, decoder, data_prep.max_source_length, data_prep.max_target_length,
                data_prep.target_char2int, data_prep.num_decoder_tokens, device)
```

## Original Features Preserved

- Character-level transliteration from English to Hindi
- Support for variable sequence lengths
- Teacher forcing during training
- Comprehensive accuracy metrics
- Model checkpointing
- Weights & Biases integration for experiment tracking

## Improvements Over Original

1. **Modularity**: Code split into logical modules
2. **Reusability**: Easy to import and use components
3. **Command Line Interface**: No need to modify code for different configurations
4. **Better Error Handling**: More robust error checking
5. **Documentation**: Clear docstrings and comments
6. **Flexibility**: Easy to extend with new features
