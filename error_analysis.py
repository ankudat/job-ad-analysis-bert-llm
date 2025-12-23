import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import sys
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = './data'
MODEL_DIR = './models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pt')
ID2LABEL_PATH = os.path.join(MODEL_DIR, 'id2label.json')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_dataset.pt')
MODEL_NAME = 'bert-base-multilingual-cased'
BATCH_SIZE = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    print("Loading resources...")
    # Load Label Mappings
    with open(ID2LABEL_PATH, 'r') as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    
    # Load Model
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Load Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Load Test Data (Use weights_only=False for local dataset)
    test_dataset = torch.load(TEST_DATA_PATH, weights_only=False)
    dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)
    
    return model, tokenizer, dataloader, id2label

def get_predictions_and_errors(model, dataloader, id2label, tokenizer):
    print("Running inference on Test Set...")
    
    true_flat = []
    pred_flat = []
    
    # Store specific examples of errors for qualitative analysis
    error_examples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device) # True labels
            
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            
            # Move to CPU for analysis
            b_input_ids = b_input_ids.cpu().numpy()
            b_labels = b_labels.cpu().numpy()
            predictions = predictions.cpu().numpy()
            
            # Iterate through batch
            for i in range(len(b_labels)):
                tokens = tokenizer.convert_ids_to_tokens(b_input_ids[i])
                label_ids = b_labels[i]
                pred_ids = predictions[i]
                
                doc_errors = []
                has_error = False
                
                for j, (tok, true_id, pred_id) in enumerate(zip(tokens, label_ids, pred_ids)):
                    # Skip padding (-100) and special tokens
                    if true_id == -100 or tok in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    true_label = id2label[str(true_id)]
                    pred_label = id2label[str(pred_id)]
                    
                    true_flat.append(true_label)
                    pred_flat.append(pred_label)
                    
                    # Check for mismatch
                    if true_label != pred_label:
                        has_error = True
                        doc_errors.append({
                            "token": tok,
                            "true": true_label,
                            "pred": pred_label,
                            "index": j
                        })
                
                # If this document had errors, save a snippet for manual inspection
                if has_error and len(error_examples) < 15: # Increased limit to 15 examples
                    # Reconstruct text snippet around the first error
                    first_err_idx = doc_errors[0]['index']
                    
                    # Increased Context Window to +/- 30 Tokens ---
                    start = max(0, first_err_idx - 30)
                    end = min(len(tokens), first_err_idx + 30)
                    context_tokens = tokens[start:end]
                    
                    error_examples.append({
                        "context": tokenizer.convert_tokens_to_string(context_tokens),
                        "errors": doc_errors[:5] # Show first 5 errors in this doc
                    })

    return true_flat, pred_flat, error_examples

def print_confusion_analysis(true_flat, pred_flat, id2label):
    labels = list(id2label.values())
    
    cm = confusion_matrix(true_flat, pred_flat, labels=labels)
    
    # Create a DataFrame for nice terminal printing
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    
    print("\n" + "="*80)
    print("PART 1: CONFUSION MATRIX (Statistical Patterns)")
    print("="*80)
    print("(Rows = True Labels, Columns = Predicted Labels)")
    print("-" * 80)
    
    # Filter for readability: Only show rows/cols with significant data
    active_labels = [lbl for lbl in labels if cm_df.loc[lbl].sum() > 0]
    cm_df_filtered = cm_df.loc[active_labels, active_labels]
    
    print(cm_df_filtered)
    
    print("\n" + "-"*80)
    print("TOP CONFUSIONS (Where does the model struggle?)")
    print("-" * 80)
    
    # Find top off-diagonal elements
    np.fill_diagonal(cm, 0) # Ignore correct predictions for this ranking
    
    pairs = []
    for i, row_label in enumerate(labels):
        for j, col_label in enumerate(labels):
            count = cm[i, j]
            if count > 0:
                pairs.append((count, row_label, col_label))
    
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for count, true_lbl, pred_lbl in pairs[:10]: # Show top 10
        print(f"{count:>4} times: True '{true_lbl}' --> Predicted '{pred_lbl}'")

def print_qualitative_examples(error_examples):
    print("\n" + "="*80)
    print("PART 2: QUALITATIVE INSPECTION (Specific Examples)")
    print("="*80)
    print("Review these snippets to find boundary issues or vocabulary triggers.")
    
    for i, ex in enumerate(error_examples):
        print(f"\n--- Example {i+1} ---")
        # Added quotes to clearly demarcate the text
        print(f"Context:\n\"{ex['context']}\"") 
        print("-" * 40)
        print("Mismatches:")
        print(f"{'Token':<20} | {'True Label':<25} | {'Predicted Label'}")
        print("-" * 60)
        for err in ex['errors']:
            clean_tok = err['token'].replace('##', '')
            print(f"{clean_tok:<20} | {err['true']:<25} | {err['pred']}")

if __name__ == "__main__":
    # Dual Logger Class to save output to file ---
    class DualLogger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open("error_analysis_report.txt", "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    # Redirect stdout to capture all print statements
    original_stdout = sys.stdout
    sys.stdout = DualLogger()
    
    try:
        model, tokenizer, dataloader, id2label = load_resources()
        true_flat, pred_flat, error_examples = get_predictions_and_errors(model, dataloader, id2label, tokenizer)
        
        print_confusion_analysis(true_flat, pred_flat, id2label)
        print_qualitative_examples(error_examples)
        print(f"\nDone. Report saved to 'error_analysis_report.txt'.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # Restore stdout 
        sys.stdout = original_stdout