import torch
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import json
from seqeval.metrics import classification_report

# --- Configuration ---
MODEL_NAME = 'bert-base-multilingual-cased'
NUM_EPOCHS = 5  # Adjust as per project spec
LEARNING_RATE = 2e-5
BATCH_SIZE = 4  # Lower if you run out of memory
GRADIENT_ACCUMULATION_STEPS = 2
DATA_DIR = './data'
MODEL_DIR = './models'
LOG_DIR = './runs/bert_experiment'
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Helper Function for Evaluation ---
def evaluate_model(model, dataloader, id2label):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=2)
            
            # Remove ignored index (special tokens)
            predictions = predictions.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            for i in range(len(predictions)):
                pred_list = []
                label_list = []
                for j in range(len(predictions[i])):
                    if label_ids[i][j] != -100:
                        pred_list.append(id2label[str(predictions[i][j])])
                        label_list.append(id2label[str(label_ids[i][j])])
                
                all_preds.append(pred_list)
                all_labels.append(label_list)
                
    return all_labels, all_preds

# --- Main Training Flow ---
if __name__ == "__main__":
    # 1. Load Data
    if not os.path.exists(os.path.join(DATA_DIR, 'train_dataset.pt')):
        print("Run preprocessing.py first!")
        exit()

    # FIX: Added weights_only=False to allow loading custom TensorDataset objects
    train_dataset = torch.load(os.path.join(DATA_DIR, 'train_dataset.pt'), weights_only=False)
    test_dataset = torch.load(os.path.join(DATA_DIR, 'test_dataset.pt'), weights_only=False)
    
    with open(os.path.join(MODEL_DIR, 'id2label.json'), 'r') as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    num_labels = len(id2label)

    # 2. Dataloaders
    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

    # 3. Model & Optimizer
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    # Use PyTorch's native AdamW
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    writer = SummaryWriter(LOG_DIR)

    # 4. Training Loop
    best_f1 = 0
    global_step = 0

    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            total_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1
                writer.add_scalar('Loss/train', total_loss, global_step)

        # Validation at end of epoch
        print(f"Epoch {epoch+1} Loss: {total_loss}")
        true_labels, pred_labels = evaluate_model(model, test_loader, id2label)
        report = classification_report(true_labels, pred_labels, output_dict=True)
        f1_score = report['weighted avg']['f1-score']
        
        print(f"Validation F1-Score: {f1_score:.4f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"New best model saved! F1: {best_f1:.4f}")

    # 5. Final Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    # Load best model
    # Note: weights_only=True is fine here because we are loading a state_dict (weights), not a complex object
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.to(device)
    
    true_labels, pred_labels = evaluate_model(model, test_loader, id2label)
    print(classification_report(true_labels, pred_labels))
    writer.close()