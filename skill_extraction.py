import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import json
import os
from google import genai
from google.genai import types

# --- Configuration ---
DATA_FILE = './data/annotated.json'
MODEL_PATH = './models/best_model.pt'
ID2LABEL_PATH = './models/id2label.json'
MODEL_NAME = 'bert-base-multilingual-cased'
OUTPUT_FILE = 'extracted_skills.json'
TARGET_LABEL = 'Fähigkeiten und Inhalte' 

# TODO: Paste your API Key here
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_available_model(client):
    """
    Selects the best available model from your account.
    """
    print("Searching for available Gemini models...")
    try:
        models_iterator = client.models.list()
        available_models = [m.name.replace('models/', '') for m in models_iterator]
        
        priorities = [
            "gemini-2.5-flash", 
            "gemini-2.0-flash", 
            "gemini-1.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash-lite"
        ]

        for p in priorities:
            if p in available_models:
                print(f"-> Selected Model: {p}")
                return p
        
        for m in available_models:
            if 'flash' in m and 'preview' not in m:
                print(f"-> Fallback Model: {m}")
                return m
                
    except Exception as e:
        print(f"Warning during model selection: {e}")
    
    return "gemini-2.5-flash"

def load_bert_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training.py first.")

    with open(ID2LABEL_PATH, 'r') as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    return model, tokenizer, id2label

def load_samples(file_path, num_samples=3):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    count = 0
    for entry in data:
        text = entry.get('data', {}).get('content_clean') or entry.get('data', {}).get('text')
        if text and len(text) > 50: 
            samples.append({"id": entry.get('id', count), "text": text})
            count += 1
            if count >= num_samples: break
    return samples

def extract_zone_text(text, model, tokenizer, id2label):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    zone_tokens = []
    for token, pred_id in zip(tokens, predictions):
        if token in ['[CLS]', '[SEP]', '[PAD]']: continue
        if id2label[str(pred_id)] == TARGET_LABEL:
            if token.startswith("##"): zone_tokens.append(token[2:])
            else: zone_tokens.append(" " + token)
    return "".join(zone_tokens).strip()

def extract_skills_with_gemini(client, model_name, zone_text):
    if not zone_text or len(zone_text) < 5: return []
    
    prompt = f"""
    You are an expert HR analyst. Extract all professional skills (hard and soft) from the text.
    Constraints: Output strictly a JSON list of strings. Phrases 2-5 words long.
    Text: "{zone_text}"
    """
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={"type": "ARRAY", "items": {"type": "STRING"}}
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini Error: {e}")
        return []

if __name__ == "__main__":
    print("Loading BERT...")
    bert_model, tokenizer, id2label = load_bert_model()
    
    if "YOUR_GEMINI" in GEMINI_API_KEY:
        print("ERROR: Please set your GEMINI_API_KEY in the script!")
    else:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            active_model_name = get_available_model(client)
        except Exception as e:
            print(f"Failed to initialize Gemini Client: {e}")
            active_model_name = None

    print(f"Loading data from {DATA_FILE}...")
    job_ads = load_samples(DATA_FILE, num_samples=3)
    results = []

    for job in job_ads:
        print(f"\n" + "="*60)
        print(f"Processing Job ID: {job['id']}")
        print("="*60)
        
        zone_text = extract_zone_text(job['text'], bert_model, tokenizer, id2label)
        
        # --- NEW: PRINT THE EXTRACTED TEXT ---
        if zone_text:
            print(f"\n[BERT] Identified Skill Zone ({len(zone_text)} chars):")
            print("-" * 40)
            print(zone_text)
            print("-" * 40 + "\n")
        else:
            print("\n[BERT] No skill zone detected.")

        skills = []
        if zone_text and active_model_name:
            print(f"Querying {active_model_name}...")
            skills = extract_skills_with_gemini(client, active_model_name, zone_text)
            # Filter 2-5 words
            skills = [s for s in skills if 2 <= len(s.split()) <= 5]
            print(f"Found Skills: {skills}")
        
        results.append({
            "job_id": job['id'],
            "skills": skills,
            "text_snippet": zone_text[:1000]
        })

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUTPUT_FILE}")