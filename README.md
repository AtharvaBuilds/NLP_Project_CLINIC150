# NLP_Project_CLINIC150
NLP for Customer Support Automation
Automated Intent Classification using RoBERTa + Streamlit

Overview
This project implements an end-to-end NLP pipeline that automatically classifies customer support queries into intent categories using a fine-tuned RoBERTa transformer model. The model achieves 97% test accuracy on the CLINC150 benchmark — exceeding the project target of 85% by 12 percentage points.
A live Streamlit web app serves as the deployment interface, allowing users to type any customer query and receive an instant predicted category with confidence scores.

Demo
Input:  "I want to cancel my subscription immediately"
Output: cancellation_intent  (confidence: 96.3%)

Input:  "What is my current account balance"
Output: check_balance  (confidence: 98.1%)

Input:  "My payment was deducted twice please refund"
Output: transaction_dispute  (confidence: 94.7%)

Project Structure
my_project/
│
├── app.py                      ← Streamlit web application
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
└── ticket_model/               ← Trained model files (download separately)
    ├── model.safetensors       ← 125M parameter RoBERTa weights (~500MB)
    ├── config.json             ← Model architecture config
    ├── tokenizer.json          ← RoBERTa BPE vocabulary (50k tokens)
    ├── tokenizer_config.json   ← Tokenizer settings
    ├── label_map.json          ← Category name → integer ID mapping
    ├── id2label.json           ← Integer ID → category name mapping
    └── metrics.json            ← Accuracy, F1, confusion matrix data

Results
MetricTargetAchievedTest Accuracy85%97.0%Macro F1High97.1%Macro PrecisionHigh97.3%Macro RecallHigh97.0%Response Time20% reduction>99% reductionTraining Time—~15 minutes (T4 GPU)

Dataset
CLINC150 — An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction
(Larson et al., EMNLP 2019)
PropertyValueTotal samples22,500Unique descriptions22,495 (99.98% unique)Intent categories150Train / Val / Test15,000 / 3,000 / 4,500LanguageEnglish
The dataset spans 10 real-world domains including banking, account management,
billing, travel, food, home automation, and general utilities — directly
relevant to customer support automation scenarios.

Why not the 200k synthetic dataset?
During development, a 200,000-row customer support dataset was evaluated.
It contained only 10 unique ticket descriptions — the rest were exact duplicates.
This produced a fake 100% accuracy through memorisation, not learning.
CLINC150 was selected for its genuine diversity and research credibility.


Model
RoBERTa (Robustly Optimized BERT Approach) by Facebook AI Research

Architecture: 12-layer transformer encoder
Parameters: 125 million
Pre-trained on: 160GB of English text
Fine-tuned for: 150-class intent classification

Why RoBERTa over BERT?
BERTRoBERTaPre-training data16GB160GB (10×)NSP taskYes (hurts classification)RemovedMasking strategyStaticDynamicTypical accuracy88-91%91-94%
Training Configuration
Epochs:          3 (early stopping, patience=2)
Batch size:      16 (train) / 32 (eval)
Learning rate:   2e-5 with linear warmup (10%) + decay
Optimizer:       AdamW (weight decay=0.01)
Max token len:   128
Hardware:        Google Colab T4 GPU
Training time:   ~15 minutes

Installation
1. Clone the repository
bashgit clone https://github.com/yourusername/nlp-customer-support.git
cd nlp-customer-support
2. Install dependencies
bashpip install -r requirements.txt
Or manually:
bashpip install streamlit transformers torch matplotlib seaborn scikit-learn numpy pandas
3. Download the trained model
Download ticket_model.zip from the Releases page and unzip it:
bashunzip ticket_model.zip -d ticket_model/
Your folder should now contain ticket_model/model.safetensors and all JSON files.
4. Run the Streamlit app
bash# Standard
streamlit run app.py

# If streamlit command not found (Windows)
python -m streamlit run app.py
Open your browser at http://localhost:8501

Note: First load takes 1-2 minutes as the 500MB model loads into memory.
Subsequent predictions are instant (cached by Streamlit).


Usage
The app has three tabs:
Live Prediction — Type any customer query or click an example button.
The model returns the predicted intent category and a confidence bar chart
showing top 10 predictions.
Model Performance — Full dashboard showing overall accuracy, macro F1,
confusion matrix (first 20 classes), per-class F1 bar chart, and summary table.
About — Project details, model specs, results vs goals, and tech stack.

Training Your Own Model
To retrain from scratch using the provided Colab notebook:

Open NLPProject_Ticket.ipynb in Google Colab
Set runtime to T4 GPU (Runtime → Change runtime type → T4 GPU)
Run all cells in order
Download ticket_model.zip from the final cell

The notebook covers:

JSON dataset loading and preprocessing
RoBERTa tokenization (max_length=128)
Custom PyTorch Dataset class
HuggingFace Trainer with early stopping
Evaluation metrics and confusion matrix
Model saving and download


Tech Stack
ComponentTechnologyModelRoBERTa (roberta-base)Training frameworkPyTorch + HuggingFace TransformersTraining platformGoogle Colab (T4 GPU)Data processingPandas, NumPyEvaluationScikit-learnVisualisationMatplotlib, SeabornWeb interfaceStreamlitLanguagePython 3.10+

Key Learnings

Data quality > data quantity — 22,500 unique real samples outperformed
200,000 synthetic duplicates. The synthetic dataset produced fake 100% accuracy
through memorisation; CLINC150 produced honest 97% through genuine learning.
Transfer learning power — RoBERTa already understands English from 160GB
of pre-training. Fine-tuning only nudges its weights slightly toward the target
task. This is why 15 minutes of training beats months of training from scratch.
fp16 caution — Mixed precision (fp16=True) caused gradient underflow on
certain Colab instances, producing 20% accuracy across all epochs. Disabling
fp16 resolved the issue immediately.
Warmup scheduler is critical — Without LR warmup, large initial gradients
from the random classification head corrupt the pre-trained weights (catastrophic
forgetting). Warmup over 10% of steps stabilises early training.
