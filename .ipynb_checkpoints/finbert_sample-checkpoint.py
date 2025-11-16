import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import torch.nn.functional as F

# ============================
# Load FinBERT models
# ============================
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
embedding_model = AutoModel.from_pretrained(MODEL_NAME)

sentiment_model.eval()
embedding_model.eval()

# ============================
# FinBERT function
# ============================
def finbert_model(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    # ---- sentiment ----
    with torch.no_grad():
        out = sentiment_model(**inputs)
        probs = F.softmax(out.logits, dim=-1)
        label_id = torch.argmax(probs).item()

    id2label = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment = id2label[label_id]

    # ---- embedding ----
    with torch.no_grad():
        emb_out = embedding_model(**inputs)
        embedding = emb_out.last_hidden_state[:, 0, :].squeeze(0)

    return sentiment, embedding.tolist()


# ============================
# Load your NEWS parquet
# ============================

INPUT_FILE = "NEWS_20240101-142500_20251101-232422.parquet"   # 修改成你的名字

df = pd.read_parquet(INPUT_FILE)

# 自动找文本列
text_col = None
for col in df.columns:
    if df[col].dtype == "object":
        text_col = col
        break

print(f"使用文本列: {text_col}")

# 取前 2 行样本
sample_df = df.head(2)

results = []

print("\n=== Running FinBERT on first 2 rows ===\n")

for i, text in enumerate(sample_df[text_col]):
    if not isinstance(text, str):
        print(f"[Row {i}] 跳过（不是文本）")
        continue
    
    sentiment, embedding = finbert_model(text)

    print(f"\n---- 样本 {i} ----")
    print("Text:", text)
    print("Sentiment:", sentiment)
    print("Embedding :", embedding)

    results.append({
        "text": text,
        "sentiment": sentiment,
        "embedding": embedding
    })

print("\n🎉 finish。")
