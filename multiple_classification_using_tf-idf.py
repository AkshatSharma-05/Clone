import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# =====================================================
# LOAD DATA
# =====================================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# =====================================================
# COMBINE TEXT
# =====================================================

train_df["text"] = (
    train_df["subject"].astype(str) + " " +
    train_df["message"].astype(str)
)

test_df["text"] = (
    test_df["subject"].astype(str) + " " +
    test_df["message"].astype(str)
)

# =====================================================
# INPUT AND TARGET
# =====================================================

X = train_df["text"]
y = train_df["ticket_category"]

# =====================================================
# TFIDF + LOGISTIC REGRESSION PIPELINE
# =====================================================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )),
    
    ("clf", LogisticRegression(
        max_iter=1000
    ))
])

# =====================================================
# TRAIN MODEL
# =====================================================

model.fit(X, y)

# =====================================================
# TEST PREDICTION
# =====================================================

predictions = model.predict(test_df["text"])

# =====================================================
# SUBMISSION FILE
# =====================================================

submission = pd.DataFrame({
    "ticket_id": test_df["ticket_id"],
    "ticket_category": predictions
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")
