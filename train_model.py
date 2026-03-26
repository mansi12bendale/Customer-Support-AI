import pandas as pd
import pickle
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("twcs.csv")

# Reduce size (important)
df = df.sample(10000)

# Keep only customer messages
df = df[df['inbound'] == True]

# Clean columns
df = df[['author_id', 'text', 'created_at']]
df = df.rename(columns={
    'author_id': 'customer_id',
    'text': 'message',
    'created_at': 'timestamp'
})

df.dropna(inplace=True)

# Categorization
def categorize(msg):
    msg = msg.lower()
    if "refund" in msg:
        return "Refund Issue"
    elif "delay" in msg or "late" in msg:
        return "Delivery Issue"
    elif "payment" in msg:
        return "Payment Issue"
    elif "broken" in msg or "damaged" in msg:
        return "Product Issue"
    elif "not working" in msg:
        return "Technical Issue"
    else:
        return "General Issue"

df["category"] = df["message"].apply(categorize)

# Sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df["sentiment"] = df["message"].apply(get_sentiment)

def label_sentiment(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

# Train model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["message"])
y = df["category"]

model = LogisticRegression()
model.fit(X, y)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
df.to_csv("tickets.csv", index=False)

print("✅ Model trained and files saved!")