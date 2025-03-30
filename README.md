STOCKWATCH 📊

Real-Time Stock Tracker with Sentiment Analysis & Price Prediction

🚀 Project Overview

STOCKWATCH is a real-time financial dashboard built with Streamlit, designed to combine stock market data, social sentiment, and machine learning-based price prediction into one clean interface.

Whether you're a retail trader, data scientist, or finance geek, STOCKWATCH provides:

Live stock tracking (via Yahoo Finance)

Sentiment monitoring from Twitter and Reddit

Trend indicators (SMA, EMA, Volume, etc.)

Predictive modeling using ML (XGBoost / LSTM)


⚖️ Key Features

📈 Live Stock Data: Real-time price updates from Yahoo Finance using yfinance

💬 Sentiment Analysis:

Twitter scraping via snscrape

Reddit data via Pushshift API

NLP-based sentiment scoring using VADER

🤖 ML Price Prediction:

Forecast next-day price using XGBoost (v1)

LSTM sequence modeling (v2)

🔄 Interactive Streamlit Dashboard:

Filtering by ticker symbol, time range, sentiment, and more

Visuals: line charts, heatmaps, bar graphs

💪 Tech Stack

Frontend: Streamlit

Data Collection: yfinance, snscrape, Pushshift API (Reddit)

NLP: NLTK, VADER, TextBlob

Modeling: XGBoost, LSTM (TensorFlow)

Visualization: Plotly, Matplotlib, Seaborn

Backend: Python, Pandas, Numpy, Scikit-learn

📆 Roadmap

Phase 1 ✅



Phase 2 ⏳



Phase 3 🚧



Phase 4 🚀



📊 Example Use Cases

Visualize Tesla ($TSLA) sentiment vs price

Track BTC price trends and predict movements

Detect social buzz before price action

🛠️ Installation

# Clone the repo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


✨ Contributions

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

If you enjoy the project, drop a star ⭐ and share your ideas!

👤 Author

Tristan TansuData Analyst & Future MLops EngineerLinkedIn | Portfolio



