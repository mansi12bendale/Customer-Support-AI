import streamlit as st
import pandas as pd
import pickle
from textblob import TextBlob

st.set_page_config(page_title="AI Dashboard", layout="wide")

st.title("🤖 AI Customer Support Insight Platform")

# Load files
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))
df = pd.read_csv("tickets.csv")

# Sidebar
menu = st.sidebar.selectbox("Menu", ["Dashboard", "Analyze Ticket"])

# -------- Dashboard --------
if menu == "Dashboard":
    st.subheader("📊 Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔥 Top Issues")
        st.bar_chart(df["category"].value_counts())

    with col2:
        st.markdown("### 😊 Sentiment")
        st.bar_chart(df["sentiment_label"].value_counts())

# -------- Analyze --------
elif menu == "Analyze Ticket":
    st.subheader("🧠 Analyze Customer Message")

    message = st.text_area("Enter message")

    if st.button("Analyze"):
        vec = vectorizer.transform([message])
        category = model.predict(vec)[0]

        sentiment_score = TextBlob(message).sentiment.polarity

        if sentiment_score < 0:
            sentiment = "Negative"
        elif sentiment_score == 0:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        # Suggested reply
        if "refund" in message.lower():
            reply = "We are processing your refund."
        elif "delay" in message.lower():
            reply = "Sorry for delay, your order is on the way."
        elif "payment" in message.lower():
            reply = "We are checking your payment."
        else:
            reply = "We are looking into your issue."

        st.success("Analysis Complete!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Category", category)
        col2.metric("Sentiment", sentiment)
        col3.metric("Reply", reply)