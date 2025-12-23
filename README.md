# Multi-Stage Job Advertisement Analysis

An advanced pipeline designed to extract structured professional skills from unstructured, multi-lingual job advertisements.  
This project combines a **BERT-based sequence labeling model** for zone identification with **Google Gemini (LLM)** for semantic information extraction.

**For full experimental details and analysis, refer to the Project Report (`project_report.pdf`) in the main branch.**

---

## 📌 Project Overview

Extracting specific skills from job descriptions is challenging due to unstructured text and diverse formatting.  
This project solves this using a two-stage approach:

1. **Zone Identification (BERT)**  
   A fine-tuned `bert-base-multilingual-cased` model segments the text to isolate the **Skills** section from other zones (e.g., Benefits, Company Description).

2. **Skill Extraction (LLM)**  
   The isolated text is processed by **Gemini 2.5 Flash** to generate a clean, JSON-formatted list of skills (2–5 word phrases).

---

## 📂 Repository Structure

```text
├── data/
│   ├── annotated.json        # Raw labeled dataset (Input)
│   ├── train_dataset.pt      # Processed training tensors
│   └── test_dataset.pt       # Processed testing tensors
├── models/
│   ├── id2label.json         # Label mapping dictionary
│   └── (best_model.pt)       # Trained weights (Download link below, too big for github)
├── error_analysis_report.txt # Output log of the qualitative analysis
├── preprocessing.py          # Data cleaning and sliding window tokenization
├── training.py               # Script to fine-tune the BERT model
├── skill_extraction.py       # Script for skill extraction (BERT + Gemini)
├── error_analysis.py         # Generates confusion matrix and error report
├── project_report.pdf        # Final Project Report (methodology, experiments, results)
└── README.md
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/job-ad-skill-extraction.git
cd job-ad-skill-extraction
```

### 2. Install Dependencies

```bash
pip install torch transformers google-genai seqeval pandas numpy scikit-learn seaborn tqdm matplotlib
```

### 3. Set Up Google Gemini API Key

To use the skill extraction pipeline, you need a valid Google GenAI API key.

1. Open `skill_extraction.py`
2. Find the line:
   ```python
   GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
   ```
3. Paste your key there

---

## 🚀 Usage Guide

### Step 1: Data Preprocessing

```bash
python preprocessing.py
```

---

### Step 2: Model Training (Optional)

```bash
python training.py
```

**Hardware:** Optimized for NVIDIA GPU (Batch size: 8)  
**Output:** Saves the model to `./models/best_model.pt`

Download pre-trained Model from Google Drive (too big for github):  
https://drive.google.com/file/d/1MnGyyfySyRG3aNLshlH1p6wRcQ7643I4/view?usp=sharing

---

### Step 3: Run Extraction Pipeline

```bash
python skill_extraction.py
```

---

### Step 4: Error Analysis

```bash
python error_analysis.py
```

---

## 📄 Final Project Report

The **Final Project Report** is available as `project_report.pdf` in the **main branch** of this repository.

---

## 💻 Hardware Environment

- **CPU:** Intel Core i9-13900KF  
- **GPU:** NVIDIA GeForce RTX 4090 (24GB VRAM)  
- **RAM:** 32GB DDR5  
