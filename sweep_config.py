import wandb
import torch
import torch.nn as nn
from data_preparation import DataPreparation
from models import Encoder, Decoder, Seq2Seq
from training_utils import train_loop


def sweep_train(sweep_config=None):
    user = "Shashank M"
    project = "Assignment_3_trial"
    display_name = "ch23s019"
    wandb.init(entity=user, project=project, name=display_name, config=sweep_config)

    config_ = wandb.config
    wandb.run.name = ("_cell_type_" + str(config_.cell_type) + "__embedding__" + str(config_.encoder_embedding_dim) + 
                     "__hidden__" + str(config_.hidden_dim) + "__attention__" + str(config_.attention) + 
                     "lr_" + str(config_.learning_rate) + "_opt_" + str(config_.optimizer) + 
                     "_epoch_" + str(config_.epochs) + "_bs_" + str(config_.batch_size))

    # Load dataset
    datapath = '/home/fmlpc/Shashank/Course_Work/CS6910-Assignment_3-main'
    dataset_func = DataPreparation(datapath)
    train_dataloader, validation_dataloader, test_dataloader = dataset_func.create_dataloaders(config_.batch_size)

    num_encoder_tokens = dataset_func.num_encoder_tokens
    hidden_dim = config_.hidden_dim
    n_layers = config_.n_layers
    encoder_embedding_dim = config_.encoder_embedding_dim
    dropout = config_.dropout
    cell_type = config_.cell_type
    decoder_embedding_dim = config_.encoder_embedding_dim
    num_decoder_tokens = dataset_func.num_decoder_tokens
    attention = config_.attention

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = Encoder(num_encoder_tokens, hidden_dim, n_layers, dropout, encoder_embedding_dim, cell_type, verbose=False)
    dec = Decoder(num_decoder_tokens, hidden_dim, n_layers, dropout, decoder_embedding_dim, cell_type, atten=attention, verbose=False)
    model = Seq2Seq(enc, dec, dataset_func.max_source_length, dataset_func.max_target_length, 
                   dataset_func.target_char2int, dataset_func.num_decoder_tokens, device)
    model = model.to(device)

    save_path = '/home/fmlpc/Shashank/Course_Work/CS6910-Assignment_3-main'
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_func.target_char2int["-PAD-"])
    train_loop(model, train_dataloader, validation_dataloader, device, criterion, config=config_, 
              save_path=save_path, clip=1)


def get_sweep_config():
    return {
        'method': 'bayes',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'epochs': {'values': [25, 30, 40]},
            'cell_type': {'values': ['GRU', 'LSTM']},
            'n_layers': {'values': [1, 2, 3]},
            'hidden_dim': {'values': [256, 400, 512, 1024]},
            'encoder_embedding_dim': {'values': [200, 256, 300, 512]},
            'dropout': {'values': [0.2, 0.4, 0.5]},
            'teacher_forcing_ratio': {'values': [0.3, 0.35, 0.4, 0.45, 0.5]},
            'learning_rate': {'min': 0.0001, 'max': 0.001},
            'optimizer': {'values': ['Adam', 'NAdam', 'RAdam', 'AdamW']},
            'batch_size': {'values': [64, 128, 256]},
            'weight_decay': {'values': [0]},
            'attention': {'values': [True]},
            'beam_width': {'values': [1, 2, 3]}
        },
    }


def run_sweep():
    sweep_config = get_sweep_config()
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="Assignment_3_start")
    wandb.agent(sweep_id, function=sweep_train, count=10)
