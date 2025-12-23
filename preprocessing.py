import pandas as pd
import torch
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import logging
import json
import os
import numpy as np
from collections import Counter

# --- Configuration & Logging ---
DATA_DIR = './data'
MODEL_DIR = './models'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Logging Setup
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
LOG_FILE_PATH = os.path.join(DATA_DIR, 'preprocessing.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE_PATH), logging.StreamHandler()]
)
logger = logging.getLogger()

def load_data(file_path):
    """Parses Label Studio JSON export."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise

    data = []
    for entry in raw_data:
        text = entry.get('data', {}).get('content_clean') or entry.get('data', {}).get('text')
        
        if 'annotations' in entry and entry['annotations']:
            annotations = entry['annotations'][0]['result']
        else:
            continue

        entities = []
        for ann in annotations:
            if ann['type'] == 'labels':
                start = ann['value']['start']
                end = ann['value']['end']
                label_val = ann['value']['labels']
                label = label_val[0] if isinstance(label_val, list) else label_val
                entities.append((start, end, label))
        
        if text:
            # We store the ID to track uniqueness if needed, though index is enough
            data.append({'text': text, 'entities': entities})
    
    return pd.DataFrame(data)

def process_with_sliding_window(tokenizer, text, entities, label2id, chunk_size=512, stride=128):
    """
    Tokenizes text using sliding window.
    Returns lists of input_ids, labels, and attention_masks for all created chunks.
    """
    tokenized_inputs = tokenizer(
        text,
        max_length=chunk_size,
        truncation=True,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    doc_input_ids = []
    doc_labels = []
    doc_attention_masks = []

    sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_inputs.pop("offset_mapping")

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_inputs["input_ids"][i]
        attention_mask = tokenized_inputs["attention_mask"][i]
        
        labels = [label2id['O']] * len(input_ids)

        for start_char, end_char, label_str in entities:
            label_id = label2id.get(label_str, label2id['O'])
            
            for token_idx, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:
                    continue
                
                if token_start >= start_char and token_end <= end_char:
                     labels[token_idx] = label_id
                elif token_start < end_char and token_end > start_char:
                     labels[token_idx] = label_id

        doc_input_ids.append(input_ids)
        doc_labels.append(labels)
        doc_attention_masks.append(attention_mask)

    return doc_input_ids, doc_labels, doc_attention_masks

def create_dataset(data_dicts):
    """Converts list of dicts to TensorDataset."""
    input_ids_list = [torch.tensor(d['input_ids']) for d in data_dicts]
    attention_masks_list = [torch.tensor(d['attention_mask']) for d in data_dicts]
    labels_list = [torch.tensor(d['labels']) for d in data_dicts]

    return TensorDataset(
        torch.stack(input_ids_list),
        torch.stack(attention_masks_list),
        torch.stack(labels_list)
    )

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting preprocessing...")
    
    # 1. Load Data
    json_path = os.path.join(DATA_DIR, 'annotated.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    
    df = load_data(json_path)
    
    # 2. Initialize Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # 3. Process Labels & Calculate Document Statistics
    unique_labels = set()
    total_annotations = 0
    
    # Counter for Document Frequency
    doc_label_counts = Counter() 

    for entities in df['entities']:
        # Get unique labels specifically in THIS document
        unique_labels_in_doc = set()
        for _, _, label in entities:
            unique_labels.add(label)
            unique_labels_in_doc.add(label)
            total_annotations += 1
        
        # Increment doc count for each label found in this doc
        for label in unique_labels_in_doc:
            doc_label_counts[label] += 1
    
    label_list = sorted(list(unique_labels))
    if 'O' not in label_list: label_list.insert(0, 'O')
    
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    
    with open(os.path.join(MODEL_DIR, 'id2label.json'), 'w') as f:
        json.dump(id2label, f)

    # 4. Tokenize with Sliding Window
    all_chunks = []
    token_label_counts = Counter() # Renamed for clarity

    logger.info("Tokenizing and creating chunks...")
    for _, row in df.iterrows():
        input_ids_list, labels_list, attn_mask_list = process_with_sliding_window(
            tokenizer, row['text'], row['entities'], label2id
        )
        
        for i in range(len(input_ids_list)):
            for l_id in labels_list[i]:
                label_str = id2label[l_id]
                if label_str != 'O':
                    token_label_counts[label_str] += 1
            
            all_chunks.append({
                'input_ids': input_ids_list[i],
                'labels': labels_list[i],
                'attention_mask': attn_mask_list[i]
            })

    # 5. Split Data
    train_chunks, test_chunks = train_test_split(all_chunks, test_size=0.2, random_state=42)
    
    # 6. Create TensorDatasets
    train_dataset = create_dataset(train_chunks)
    test_dataset = create_dataset(test_chunks)

    # 7. Save
    torch.save(train_dataset, os.path.join(DATA_DIR, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(DATA_DIR, 'test_dataset.pt'))
    
    # --- DETAILED REPORT ---
    print("\n" + "="*50)
    print("           DATASET PROCESSING REPORT             ")
    print("           (Sliding Window Applied)              ")
    print("="*50)
    print(f"Original Documents:      {len(df)}")
    print(f"Total Chunks Generated:  {len(all_chunks)}")
    print("-" * 50)
    print(f"Training Chunks:         {len(train_dataset)}")
    print(f"Test Chunks:             {len(test_dataset)}")
    print("-" * 50)
    print(f"{'LABEL':<30} | {'DOCS (Count)':<15} | {'TOKENS (Sum)':<15}")
    print("-" * 50)
    
    # Combine stats for display
    # Sort by Document Count first
    for label, doc_count in doc_label_counts.most_common():
        token_count = token_label_counts[label]
        print(f"{label:<30} | {doc_count:<15} | {token_count:<15}")
        
    print("="*50 + "\n")
    
    logger.info("Preprocessing complete.")