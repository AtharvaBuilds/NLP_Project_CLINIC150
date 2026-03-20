import streamlit as st
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support Classifier",
    page_icon="🎫",
    layout="wide"
)

# ── Model path — update this to your exact path ───────────────
MODEL_PATH = r"D:\ml lab exxefiles\NLP\my_project\ticket_model"

# ── Load model cached ─────────────────────────────────────────
@st.cache_resource
def load_model():
    st.write("Loading model... please wait 1-2 minutes on first run.")

    # Check path exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model folder not found at: {MODEL_PATH}")
        st.stop()

    # Load JSON files
    with open(os.path.join(MODEL_PATH, "id2label.json")) as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(MODEL_PATH, "label_map.json")) as f:
        label_map = json.load(f)

    # Load metrics if exists
    metrics_path = os.path.join(MODEL_PATH, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = None

    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True    # ← reduces RAM usage on load
    )
    model.eval()
    model.to("cpu")

    return model, tokenizer, id2label, label_map, metrics

# ── Load ──────────────────────────────────────────────────────
with st.spinner("Loading model — first load takes 1-2 mins..."):
    model, tokenizer, id2label, label_map, metrics = load_model()

label_names = list(id2label.values())

# ── Header ────────────────────────────────────────────────────
st.title("🎫 Customer Support Ticket Classifier")
st.caption("NLP Project — RoBERTa fine-tuned on CLINC150")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Live Prediction",
    "Model Performance",
    "About"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Predict Ticket Category")

    # Example buttons
    st.markdown("**Try an example:**")
    examples = [
        "I want to cancel my subscription",
        "My payment was deducted twice please refund",
        "The app keeps crashing on my phone",
        "What is included in the premium plan",
        "My account is locked and I cannot login",
        "I need to update my billing details",
    ]

    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(example, key=f"ex_{i}"):
            st.session_state.ticket_input = example

    # Text input
    ticket_text = st.text_area(
        "Enter ticket text:",
        value=st.session_state.get("ticket_input", ""),
        height=120,
        placeholder="Type your customer support query here..."
    )

    if st.button("Predict", type="primary"):
        if ticket_text.strip():
            with st.spinner("Predicting..."):
                # Tokenize
                inputs = tokenizer(
                    ticket_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )

                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.softmax(
                    outputs.logits, dim=1)[0].cpu().numpy()
                pred_id = int(np.argmax(probs))
                pred_label = id2label[pred_id]
                confidence = float(probs[pred_id]) * 100

            # Show result
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"**Predicted: {pred_label}**")
                st.metric("Confidence", f"{confidence:.1f}%")

                # Top 5 predictions
                st.markdown("**Top predictions:**")
                top5_idx = np.argsort(probs)[::-1][:5]
                for idx in top5_idx:
                    bar_val = float(probs[idx])
                    st.write(f"`{id2label[idx]}` — {bar_val*100:.1f}%")
                    st.progress(bar_val)

            with col2:
                # Probability chart
                fig, ax = plt.subplots(figsize=(5, 4))
                top10_idx = np.argsort(probs)[::-1][:10]
                top10_labels = [id2label[i] for i in top10_idx]
                top10_probs = probs[top10_idx] * 100
                colors = ["#4CAF50" if i == pred_id else "#2196F3"
                          for i in top10_idx]
                ax.barh(top10_labels[::-1],
                        top10_probs[::-1],
                        color=colors[::-1])
                ax.set_xlabel("Probability (%)")
                ax.set_title("Top 10 Predictions")
                ax.set_xlim(0, 100)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.warning("Please enter some text first!")

# ════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Dashboard")

    if metrics is None:
        st.warning("metrics.json not found in ticket_model folder.")
        st.info("Re-run the final Colab cell to regenerate metrics.json")
    else:
        # Top metrics
        accuracy = metrics["accuracy"]
        report = metrics["classification_report"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy",
                    f"{accuracy*100:.2f}%",
                    delta="beats 85% target")
        col2.metric("Macro F1",
                    f"{report['macro avg']['f1-score']*100:.2f}%")
        col3.metric("Macro Precision",
                    f"{report['macro avg']['precision']*100:.2f}%")
        col4.metric("Macro Recall",
                    f"{report['macro avg']['recall']*100:.2f}%")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**Confusion Matrix**")
            cm = np.array(metrics["confusion_matrix"])

            # Show only first 20 classes if too many
            n = min(20, len(label_names))
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                cm[:n, :n],
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=label_names[:n],
                yticklabels=label_names[:n],
                ax=ax
            )
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            ax.set_title(
                f"Confusion Matrix (first {n} classes)",
                fontweight='bold'
            )
            plt.xticks(rotation=45, ha='right', fontsize=7)
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown("**Per-Class F1 Scores**")

            # Per class F1 bar chart
            f1_scores = []
            class_labels = []
            for label in label_names:
                if label in report:
                    f1_scores.append(
                        report[label]['f1-score'] * 100)
                    class_labels.append(label)

            fig2, ax2 = plt.subplots(figsize=(6, 8))
            ax2.barh(class_labels, f1_scores, color='steelblue')
            ax2.set_xlabel("F1 Score (%)")
            ax2.set_title("Per-Class F1", fontweight='bold')
            ax2.set_xlim(0, 105)
            ax2.axvline(x=85, color='red',
                        linestyle='--', label='85% target')
            ax2.legend()
            plt.yticks(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

            # Summary table
            st.markdown("**Summary Statistics**")
            st.dataframe({
                "Metric": ["Accuracy", "Macro F1",
                           "Macro Precision", "Macro Recall"],
                "Score": [
                    f"{accuracy*100:.2f}%",
                    f"{report['macro avg']['f1-score']*100:.2f}%",
                    f"{report['macro avg']['precision']*100:.2f}%",
                    f"{report['macro avg']['recall']*100:.2f}%"
                ]
            }, hide_index=True, use_container_width=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    ### NLP for Customer Support Automation

    **Problem:** Manually categorising customer support tickets
    is slow and inefficient.

    **Solution:** Fine-tune a RoBERTa transformer model to
    automatically classify customer queries in under 1 second.

    ---

    ### Model Details
    | Property | Value |
    |----------|-------|
    | Architecture | RoBERTa (roberta-base) |
    | Parameters | 125 million |
    | Dataset | CLINC150 (22,500 samples) |
    | Categories | 150 intent classes |
    | Epochs | 3 (with early stopping) |
    | Optimizer | AdamW |
    | Learning rate | 2e-5 with warmup |
    | Test Accuracy | 97% |

    ---

    ### Results vs Goals
    | Goal | Target | Achieved |
    |------|--------|---------|
    | Classification accuracy | 85% | 97% |
    | Response time reduction | 20% | 99% |
    | Category precision | High | 97% macro |

    ---

    ### Tech Stack
    `Python` `PyTorch` `HuggingFace Transformers`
    `Streamlit` `Scikit-learn` `Matplotlib` `Seaborn`
    """)