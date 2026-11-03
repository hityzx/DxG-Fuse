import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tqdm import tqdm 
import datetime

from dataloader import create_dataloader, get_image_transform
from model import dxg

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value):
    """Resolve local paths relative to the project, independent of the cwd."""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else PROJECT_ROOT / path)

def log_message(log_file_path, message):
    """Appends a message to the specified log file."""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    except Exception as e:
        print(f"Error writing to log file {log_file_path}: {e}")

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_epoch(model, data_loader, loss_fn_primary, loss_fn_aux, optimizer, device, scheduler, aux_weights):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    total_primary_loss = 0.0
    total_aux_loss = 0.0

    pbar = tqdm(data_loader, desc="Training Epoch")
    for batch in pbar:
        optimizer.zero_grad()

        input_ids_content = batch['content_ids'].to(device)
        attention_mask_content = batch['content_mask'].to(device)
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        cs_rationale_ids = batch['cs_rationale_ids'].to(device)
        cs_rationale_mask = batch['cs_rationale_mask'].to(device)
        cs_pred = batch['cs_pred'].to(device)
        td_rationale_ids = batch['td_rationale_ids'].to(device)
        td_rationale_mask = batch['td_rationale_mask'].to(device)
        td_pred = batch['td_pred'].to(device)
        img_rationale_ids = batch['img_rationale_ids'].to(device)
        img_rationale_mask = batch['img_rationale_mask'].to(device)
        img_pred = batch['img_pred'].to(device)
        img_desc_ids = batch['img_desc_ids'].to(device)
        img_desc_mask = batch['img_desc_mask'].to(device)


        outputs = model(
            content_ids=input_ids_content, content_mask=attention_mask_content,
            image=images,
            cs_rationale_ids=cs_rationale_ids, cs_rationale_mask=cs_rationale_mask, cs_pred=cs_pred,
            td_rationale_ids=td_rationale_ids, td_rationale_mask=td_rationale_mask, td_pred=td_pred,
            img_rationale_ids=img_rationale_ids, img_rationale_mask=img_rationale_mask, img_pred=img_pred,
            img_desc_ids=img_desc_ids, img_desc_mask=img_desc_mask
        )

        logits_primary = outputs['logits_primary']
        loss_primary = loss_fn_primary(logits_primary.squeeze(-1), labels.float())

        valid_cs_indices = cs_pred >= 0
        valid_td_indices = td_pred >= 0
        valid_img_indices = img_pred >= 0

        loss_aux_cont = loss_fn_aux(outputs['logits_aux_cont'][valid_cs_indices], cs_pred[valid_cs_indices]) if valid_cs_indices.any() else torch.tensor(0.0, device=device)
        loss_aux_senti = loss_fn_aux(outputs['logits_aux_senti'][valid_td_indices], td_pred[valid_td_indices]) if valid_td_indices.any() else torch.tensor(0.0, device=device)
        loss_aux_cons = loss_fn_aux(outputs['logits_aux_cons'][valid_img_indices], img_pred[valid_img_indices]) if valid_img_indices.any() else torch.tensor(0.0, device=device)


        w_cont, w_senti, w_cons = aux_weights
        loss_aux = w_cont * loss_aux_cont + w_senti * loss_aux_senti + w_cons * loss_aux_cons

        total_loss_batch = loss_primary + loss_aux

        total_loss_batch.backward()

        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += total_loss_batch.item()
        total_primary_loss += loss_primary.item()
        if valid_cs_indices.any() or valid_td_indices.any() or valid_img_indices.any():
             total_aux_loss += loss_aux.item()

        pbar.set_postfix(loss=total_loss_batch.item())

    num_batches = len(data_loader) if len(data_loader) > 0 else 1
    avg_loss = total_loss / num_batches
    avg_primary_loss = total_primary_loss / num_batches
    avg_aux_loss = total_aux_loss / num_batches
    return avg_loss, avg_primary_loss, avg_aux_loss


def evaluate(model, data_loader, loss_fn_primary, loss_fn_aux, device, aux_weights):
    """Evaluates the model on the test/validation set."""
    model.eval()
    total_eval_loss = 0.0
    all_labels = []
    all_preds_primary = []
    all_probs_primary = []

    pbar = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            input_ids_content = batch['content_ids'].to(device)
            attention_mask_content = batch['content_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            cs_rationale_ids = batch['cs_rationale_ids'].to(device)
            cs_rationale_mask = batch['cs_rationale_mask'].to(device)
            cs_pred = batch['cs_pred'].to(device)
            td_rationale_ids = batch['td_rationale_ids'].to(device)
            td_rationale_mask = batch['td_rationale_mask'].to(device)
            td_pred = batch['td_pred'].to(device)
            img_rationale_ids = batch['img_rationale_ids'].to(device)
            img_rationale_mask = batch['img_rationale_mask'].to(device)
            img_pred = batch['img_pred'].to(device)
            img_desc_ids = batch['img_desc_ids'].to(device)
            img_desc_mask = batch['img_desc_mask'].to(device)

            outputs = model(
                content_ids=input_ids_content, content_mask=attention_mask_content,
                image=images,
                cs_rationale_ids=cs_rationale_ids, cs_rationale_mask=cs_rationale_mask, cs_pred=cs_pred,
                td_rationale_ids=td_rationale_ids, td_rationale_mask=td_rationale_mask, td_pred=td_pred,
                img_rationale_ids=img_rationale_ids, img_rationale_mask=img_rationale_mask, img_pred=img_pred,
                img_desc_ids=img_desc_ids, img_desc_mask=img_desc_mask
            )

            logits_primary = outputs['logits_primary']
            valid_label_indices = labels >= 0
            if valid_label_indices.any():
                loss_primary = loss_fn_primary(logits_primary.squeeze(-1)[valid_label_indices], labels[valid_label_indices].float())
                total_eval_loss += loss_primary.item()

            probs = torch.sigmoid(logits_primary.squeeze(-1)[valid_label_indices])
            preds = (probs > 0.5).long()

            all_labels.extend(labels[valid_label_indices].cpu().numpy())
            all_preds_primary.extend(preds.cpu().numpy())
            all_probs_primary.extend(probs.cpu().numpy())

    num_eval_batches = len(data_loader) if len(data_loader) > 0 else 1
    avg_eval_loss = total_eval_loss / num_eval_batches if total_eval_loss > 0 else 0.0

    accuracy = 0.0
    auc = 0.0
    f1_real = 0.0
    f1_fake = 0.0
    f1_macro = 0.0

    if all_labels: 
        accuracy = accuracy_score(all_labels, all_preds_primary)
        try:
            if len(np.unique(all_labels)) > 1:
                 auc = roc_auc_score(all_labels, all_probs_primary)
            else:
                 print(f"Warning: Only one class present in eval labels ({np.unique(all_labels)}). AUC is not defined. Setting AUC to 0.0.")
                 auc = 0.0
        except ValueError as e:
            print(f"Warning: Could not calculate AUC ({e}). Setting AUC to 0.")
            auc = 0.0

        f1_scores_per_class = f1_score(all_labels, all_preds_primary, average=None, labels=[0, 1], zero_division=0)
        f1_real = f1_scores_per_class[0] if len(f1_scores_per_class) > 0 else 0.0
        f1_fake = f1_scores_per_class[1] if len(f1_scores_per_class) > 1 else 0.0
        f1_macro = f1_score(all_labels, all_preds_primary, average='macro', zero_division=0)
    else:
        print("Warning: No valid labels found during evaluation. Metrics calculation skipped.")


    metrics = {
        'loss': avg_eval_loss,
        'accuracy': accuracy,
        'auc': auc,
        'f1_real': f1_real,
        'f1_fake': f1_fake,
        'f1_macro': f1_macro
    }
    return metrics


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    train_parts = os.path.normpath(args.train_json).split(os.sep)
    dataset_name = train_parts[-3] if len(train_parts) >= 3 else "unknown"
    model_name = train_parts[-2] if len(train_parts) >= 2 else "unknown"
    run_tag = f"{model_name}_{dataset_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_file_path = os.path.join(args.output_dir, f"training_log_{run_tag}.txt")
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_message(log_file_path, f"\n===== Training Run Started: {start_time} =====")

    log_message(log_file_path, "Hyperparameters:")
    args_dict = vars(args)
    for key, value in args_dict.items():
        log_message(log_file_path, f"- {key}: {value}")
    log_message(log_file_path, "-" * 20)

    set_seed(args.seed)
    log_message(log_file_path, f"Random Seed: {args.seed}")

    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
        log_message(log_file_path, f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device("cpu")
        log_message(log_file_path, "Using CPU")

    print(f"Loading tokenizer: {args.bert_model}")
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    image_transform = get_image_transform(image_size=args.image_size)

    print("Creating DataLoaders...")
    train_loader = create_dataloader(
        json_path=args.train_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = create_dataloader(
        json_path=args.val_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = create_dataloader(
        json_path=args.test_json,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    if train_loader is None or val_loader is None or test_loader is None:
        err_msg = "Error creating dataloaders. Exiting."
        print(err_msg)
        log_message(log_file_path, err_msg)
        return

    model = dxg(
        bert_model_name=args.bert_model,
        vit_model_name=args.vit_model,
        judgment_embed_dim=args.judgment_embed_dim,
        perspective_dim=args.perspective_dim,
        enhanced_img_dim=args.enhanced_img_dim,
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        dropout_rate=args.dropout
    )
    model.to(device)

    log_message(log_file_path, "--- Applying Layer Freezing ---")

    if args.unfreeze_bert_layers > -1:
        freeze_info = f"Freezing BERT layers except the last {args.unfreeze_bert_layers} encoder layers and pooler."
        print(freeze_info)
        log_message(log_file_path, freeze_info)
        for param in model.bert.parameters():
            param.requires_grad = False

        if args.unfreeze_bert_layers > 0:
            num_bert_layers = len(model.bert.encoder.layer)
            for i in range(args.unfreeze_bert_layers):
                layer_index_from_end = i + 1
                if layer_index_from_end <= num_bert_layers:
                    actual_layer_index = num_bert_layers - layer_index_from_end
                    print(f"Unfreezing BERT layer index: {actual_layer_index} (from end: {layer_index_from_end})")
                    for param in model.bert.encoder.layer[actual_layer_index].parameters():
                        param.requires_grad = True
                else:
                    print(f"Warning: Requested unfreeze layer index {layer_index_from_end} (from end) out of bounds for BERT ({num_bert_layers} layers).")
                    break

        if hasattr(model.bert, 'pooler') and model.bert.pooler is not None:
             print("Unfreezing BERT pooler.")
             for param in model.bert.pooler.parameters():
                 param.requires_grad = True
        else:
             print("BERT model has no pooler or it's None.")

    else:
        full_ft_info = "Full fine-tuning enabled for BERT (all layers unfrozen)."
        print(full_ft_info)
        log_message(log_file_path, full_ft_info)
        for param in model.bert.parameters():
             param.requires_grad = True

    if args.unfreeze_vit_layers > -1:
        freeze_info_vit = f"Freezing ViT layers except the last {args.unfreeze_vit_layers} encoder blocks, layernorm, and pooler."
        print(freeze_info_vit)
        log_message(log_file_path, freeze_info_vit)
        for param in model.vit.parameters():
            param.requires_grad = False

        if args.unfreeze_vit_layers > 0:
            num_vit_layers = len(model.vit.layers)
            for i in range(args.unfreeze_vit_layers):
                layer_index_from_end = i + 1
                if layer_index_from_end <= num_vit_layers:
                    actual_layer_index = num_vit_layers - layer_index_from_end
                    print(f"Unfreezing ViT layer (block) index: {actual_layer_index} (from end: {layer_index_from_end})")
                    for param in model.vit.layers[actual_layer_index].parameters():
                        param.requires_grad = True
                else:
                    print(f"Warning: Requested unfreeze layer index {layer_index_from_end} (from end) out of bounds for ViT ({num_vit_layers} layers).")
                    break

        if hasattr(model.vit, 'layernorm') and model.vit.layernorm is not None:
             print("Unfreezing ViT final layernorm.")
             for param in model.vit.layernorm.parameters():
                  param.requires_grad = True

        if hasattr(model.vit, 'pooler') and model.vit.pooler is not None:
             print("Unfreezing ViT pooler.")
             for param in model.vit.pooler.parameters():
                 param.requires_grad = True
        else:
            print("ViT model has no pooler or it's None.")

    else:
        full_ft_info_vit = "Full fine-tuning enabled for ViT (all layers unfrozen)."
        print(full_ft_info_vit)
        log_message(log_file_path, full_ft_info_vit)
        for param in model.vit.parameters():
             param.requires_grad = True


    loss_fn_primary = nn.BCEWithLogitsLoss()
    loss_fn_aux = nn.CrossEntropyLoss()

    print("Creating optimizer for trainable parameters...")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    trainable_info = f"Number of trainable parameters: {num_trainable:,} / Total parameters: {num_total:,}"
    print(trainable_info)
    log_message(log_file_path, trainable_info)

    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_accuracy = float('-inf')
    best_epoch_metrics = {}
    best_epoch = -1
    epochs_no_improve = 0
    best_model_path_logged = None

    print("\n--- Starting Training ---")
    log_message(log_file_path, "--- Training Started ---")
    log_message(log_file_path, "Epoch | Train Loss | Train Pri Loss | Train Aux Loss | Validation Loss | Val Acc | Validation AUC | Validation F1-Real | Validation F1-Fake | Best Acc")
    log_message(log_file_path, "-" * 120)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_primary_loss, train_aux_loss = train_epoch(
            model, train_loader, loss_fn_primary, loss_fn_aux, optimizer, device, scheduler, args.aux_weights
        )
        train_log_console = f"Epoch {epoch+1} Train Results: Avg Loss={train_loss:.4f}, Primary Loss={train_primary_loss:.4f}, Aux Loss={train_aux_loss:.4f}"
        print(train_log_console)

        val_metrics = evaluate(
            model, val_loader, loss_fn_primary, loss_fn_aux, device, args.aux_weights
        )
        val_log_console = (f"Epoch {epoch+1} Validation Results:  Loss={val_metrics['loss']:.4f}, "
                           f"Acc={val_metrics['accuracy']:.4f}, AUC={val_metrics['auc']:.4f}, "
                           f"F1-Real={val_metrics['f1_real']:.4f}, F1-Fake={val_metrics['f1_fake']:.4f}, "
                           f"F1-Macro={val_metrics['f1_macro']:.4f}")
        print(val_log_console)

        log_line = (f"{epoch+1:^5} | {train_loss:^10.4f} | {train_primary_loss:^14.4f} | {train_aux_loss:^14.4f} | "
                    f"{val_metrics['loss']:^9.4f} | {val_metrics['accuracy']:^8.4f} | {val_metrics['auc']:^8.4f} | "
                    f"{val_metrics['f1_real']:^12.4f} | {val_metrics['f1_fake']:^12.4f} | {max(best_accuracy, val_metrics['accuracy']):^8.4f}")
        log_message(log_file_path, log_line)

        current_accuracy = val_metrics['accuracy']
        if current_accuracy > best_accuracy:
            previous_best_path = best_model_path_logged
            best_accuracy = current_accuracy
            best_epoch_metrics = val_metrics
            best_epoch = epoch + 1
            epochs_no_improve = 0

            current_best_model_name = f"{run_tag}_epoch{best_epoch}_acc{best_accuracy:.4f}.pt"
            current_best_model_path = os.path.join(args.output_dir, current_best_model_name)

            try:
                torch.save(model.state_dict(), current_best_model_path)
                best_model_path_logged = current_best_model_path
                best_update_log = f"--- Epoch {epoch+1}: New Best Model Saved! Validation Acc: {best_accuracy:.4f} Path: {current_best_model_name} ---"
                print(best_update_log)
                log_message(log_file_path, best_update_log)
                if previous_best_path and os.path.exists(previous_best_path) and previous_best_path != current_best_model_path:
                    try:
                        os.remove(previous_best_path)
                        removed_msg = f"Removed previous best model: {os.path.basename(previous_best_path)}"
                        print(removed_msg)
                        log_message(log_file_path, f"-> {removed_msg}")
                    except OSError as e:
                        error_msg = f"Error removing previous best model {previous_best_path}: {e}"
                        print(error_msg)
                        log_message(log_file_path, f"-> {error_msg}")
            except Exception as e:
                save_error_msg = f"Error saving model {current_best_model_path}: {e}"
                print(save_error_msg)
                log_message(log_file_path, f"-> {save_error_msg}")
                raise


        else:
            epochs_no_improve += 1
            no_improve_msg = f"Validation accuracy did not improve. Patience: {epochs_no_improve}/{args.patience}"
            print(no_improve_msg)
            log_message(log_file_path, f"-> Epoch {epoch+1}: {no_improve_msg}")

        if best_epoch_metrics:
             print(f"--> Best Validation Results So Far (Epoch {best_epoch}): Acc={best_epoch_metrics['accuracy']:.4f}, AUC={best_epoch_metrics['auc']:.4f}, "
                   f"F1-Real={best_epoch_metrics['f1_real']:.4f}, F1-Fake={best_epoch_metrics['f1_fake']:.4f}")

        if epochs_no_improve >= args.patience:
            stop_message = f"\nEarly stopping triggered after {args.patience} epochs without improvement."
            print(stop_message)
            log_message(log_file_path, stop_message)
            break

    print("\n--- Training Finished ---")
    log_message(log_file_path, "--- Training Finished ---")
    if best_model_path_logged and os.path.exists(best_model_path_logged):
        final_log1 = f"Best model saved at: {os.path.basename(best_model_path_logged)} (Epoch {best_epoch})"
        print(final_log1)
        log_message(log_file_path, final_log1)
        final_log2 = "Final Best Validation Metrics:"
        print(final_log2)
        log_message(log_file_path, final_log2)
        for key, value in best_epoch_metrics.items():
            metric_log = f"- {key}: {value:.4f}"
            print(metric_log)
            log_message(log_file_path, metric_log)
    elif best_epoch_metrics:
         final_log_nosave = "Model file for best epoch not found, but best metrics were:"
         print(final_log_nosave)
         log_message(log_file_path, final_log_nosave)
         for key, value in best_epoch_metrics.items():
            metric_log = f"- {key}: {value:.4f}"
            print(metric_log)
            log_message(log_file_path, metric_log)
    else:
        final_log_empty = "No improvement detected during training, or training stopped early. No best model saved."
        print(final_log_empty)
        log_message(log_file_path, final_log_empty)

    if not best_model_path_logged or not os.path.isfile(best_model_path_logged):
        error_message = "Best validation checkpoint is unavailable; cannot run final testing."
        log_message(log_file_path, error_message)
        raise RuntimeError(error_message)

    checkpoint_message = f"Loading best validation checkpoint for testing: {best_model_path_logged} (Epoch {best_epoch})"
    print(checkpoint_message)
    log_message(log_file_path, checkpoint_message)
    model.load_state_dict(torch.load(best_model_path_logged, map_location='cpu', weights_only=True))
    final_test_metrics = evaluate(
        model, test_loader, loss_fn_primary, loss_fn_aux, device, args.aux_weights
    )
    final_test_header = f"Final Test Metrics (best validation epoch {best_epoch}):"
    print(final_test_header)
    log_message(log_file_path, final_test_header)
    for key, value in final_test_metrics.items():
        metric_log = f"- {key}: {value:.4f}"
        print(metric_log)
        log_message(log_file_path, metric_log)

    log_message(log_file_path, "=" * 25 + " Run End " + "=" * 25 + "\n")

    if os.path.exists(log_file_path):
        new_name = f"training_log_{run_tag}_acc{best_accuracy:.4f}.txt"
        new_path = os.path.join(args.output_dir, new_name)
        try:
            os.rename(log_file_path, new_path)
            print(f"Log saved to: {new_name}")
        except OSError as e:
            print(f"Error renaming log file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dxg model for Fake News Detection")

    parser.add_argument('--train_json', type=str, default='./data/twitter/train.json', help='Path to the training JSON file')
    parser.add_argument('--val_json', type=str, default='./data/twitter/val.json', help='Path to the validation JSON file')
    parser.add_argument('--test_json', type=str, default='./data/twitter/test.json', help='Path to the test JSON file')
    parser.add_argument('--output_dir', type=str, default='./saved_models', help='Directory to save the best model and log file')

    parser.add_argument('--bert_model', type=str, default='./pretrained/bert-base-uncased', help='Local pre-trained BERT directory')
    parser.add_argument('--vit_model', type=str, default='./pretrained/vit', help='Local pre-trained ViT directory')

    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW optimizer')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length for BERT')
    parser.add_argument('--image_size', type=int, default=224, help='Image size (height and width)')
    parser.add_argument('--aux_weights', type=float, nargs=3, default=[0.2, 0.2, 0.2], help='Weights for the 3 auxiliary losses (content, sentiment, consistency)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate in the model')

    parser.add_argument('--unfreeze_bert_layers', type=int, default=-1, help='Number of last BERT encoder layers to unfreeze (0 to freeze all encoder layers, -1 to unfreeze all)')
    parser.add_argument('--unfreeze_vit_layers', type=int, default=-1, help='Number of last ViT encoder blocks to unfreeze (0 to freeze all encoder blocks, -1 to unfreeze all)')

    parser.add_argument('--judgment_embed_dim', type=int, default=64)
    parser.add_argument('--perspective_dim', type=int, default=256)
    parser.add_argument('--enhanced_img_dim', type=int, default=512)
    parser.add_argument('--transformer_d_model', type=int, default=768)
    parser.add_argument('--transformer_nhead', type=int, default=8)
    parser.add_argument('--transformer_num_layers', type=int, default=2)
    parser.add_argument('--transformer_dim_feedforward', type=int, default=2048)


    parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be at least 1")

    for name in ('train_json', 'val_json', 'test_json', 'output_dir', 'bert_model', 'vit_model'):
        setattr(args, name, project_path(getattr(args, name)))
    for name in ('train_json', 'val_json', 'test_json'):
        if not Path(getattr(args, name)).is_file():
            parser.error(f'{name} file not found: {getattr(args, name)}')
    for name in ('bert_model', 'vit_model'):
        if not Path(getattr(args, name)).is_dir():
            parser.error(f'{name} directory not found: {getattr(args, name)}')

    if len(args.aux_weights) != 3:
        raise ValueError("aux_weights must contain exactly three values.")

    main(args)

