import torch
import torch.nn as nn
import os


def accuracy_calc(target, output, train=True):
    target = target.transpose(0, 1)
    num_correct = 0
    batch_size = target.shape[0]
    target_indices = (target == 1).nonzero()[:, 1]

    assert batch_size == len(target_indices)

    if train:
        output = output.argmax(2)
        output = output.transpose(0, 1)
        for seq, i in zip(range(batch_size), target_indices):
            if torch.all(output[seq, :i + 1] == target[seq, :i + 1]):
                num_correct += 1
    else:
        for seq, i in zip(range(batch_size), target_indices):
            if torch.all(torch.tensor(output[seq][0]).to(target.device) == target[seq, :i + 1]):
                num_correct += 1

    return num_correct, batch_size


def train(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0.0
    total_no_correct = 0
    total_samples = 0

    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        src, trg = src.to(device), trg.to(device)
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio, to_train=True).to(device)
        trg = trg.transpose(0, 1)
        trg = trg.argmax(2)
        num_correct, num_samples = accuracy_calc(trg, output, train=True)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        total_no_correct += num_correct
        total_samples += num_samples

    return epoch_loss / len(iterator), total_no_correct / total_samples


def evaluate(model, iterator, criterion, beam_width, device):
    model.eval()
    epoch_loss = 0
    total_no_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0, to_train=True).to(device)
            trg = trg.transpose(0, 1)
            trg = trg.argmax(2)
            num_correct, num_samples = accuracy_calc(trg, output, train=True)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            total_no_correct += num_correct
            total_samples += num_samples

    return epoch_loss / len(iterator), total_no_correct / total_samples


def train_loop(model, train_dataloader, validation_dataloader, device, criterion, config, save_path, clip=1, sweep=True):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'NAdam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    epoch_loss_train = []
    epoch_loss_val = []
    epoch_accuracy_train = []
    epoch_accuracy_val = []

    for epoch in range(1, config.epochs + 1):
        print(f"\nEPOCH: {epoch}")

        train_loss, train_accuracy = train(model, train_dataloader, optimizer, criterion, clip, config.teacher_forcing_ratio, device)
        val_loss, val_accuracy = evaluate(model, validation_dataloader, criterion, config.beam_width, device)

        if sweep:
            import wandb
            wandb.log({"validation_accuracy": val_accuracy, "validation_loss": val_loss, 
                      "training_accuracy": train_accuracy, "training_loss": train_loss, "epochs": epoch})

        epoch_loss_train.append(train_loss)
        epoch_loss_val.append(val_loss)
        epoch_accuracy_train.append(train_accuracy)
        epoch_accuracy_val.append(val_accuracy)

        print(f"TRAINING LOSS: {train_loss}")
        print(f"TRAINING ACCURACY: {train_accuracy}")
        print(f"VALIDATION LOSS: {val_loss}")
        print(f"VALIDATION ACCURACY: {val_accuracy}")

    torch.save(model, os.path.join(save_path, 'model.pth'))
    if not sweep:
        return epoch_loss_train, epoch_loss_val, epoch_accuracy_train, epoch_accuracy_val


def evaluate_with_beam_search(model, iterator, beam_width, device):
    """Evaluate model using beam search decoding"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(device)
            trg = trg.to(device)
            
            # Get beam search predictions
            predictions = model.beam_search(src, beam_width=beam_width)
            
            # Convert targets to token indices
            trg = trg.transpose(0, 1).argmax(2)  # [seq_len, batch_size]
            
            batch_size = trg.size(1)
            for b in range(batch_size):
                # Find end token in target
                target_seq = trg[:, b]
                try:
                    end_idx = (target_seq == model.target_chr2int['\n']).nonzero()[0].item()
                    target_tokens = target_seq[1:end_idx].cpu().tolist()  # Skip start token
                except:
                    target_tokens = target_seq[1:].cpu().tolist()
                
                # Compare with prediction
                pred_tokens = predictions[b]
                if len(pred_tokens) > 0 and pred_tokens[-1] == model.target_chr2int['\n']:
                    pred_tokens = pred_tokens[:-1]  # Remove end token
                
                if pred_tokens == target_tokens:
                    total_correct += 1
                total_samples += 1
    
    return total_correct / total_samples
