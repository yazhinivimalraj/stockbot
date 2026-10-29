Stockbot – Commodity Price Direction Neural Network + Chatbot

A project that trains a neural network on Indian commodity market data and provides a chatbot (Stockbot) that predicts whether a commodity's price is likely to increase or decrease.

Dataset: vishnun0027/indian-market-historical-ohlcv (commodities configuration) from Hugging Face.

Disclaimer: Price direction is very hard to predict. This is a learning project, not financial advice.

Project structure
project/
├── script4.py         # training module – trains the neural network and saves it
├── script5.py         # Stockbot – the chatbot that loads the saved model
├── saved_model1/      # the trained model (created by script4.py)
├── requirements.txt   # Python dependencies
└── README.md
Setup
bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
1. Train the model
bash
python script4.py

This loads the commodities dataset, builds the input features and the Increase/Decrease target (next-day close vs today's close), trains the neural network, and saves it to saved_model1/.

2. Run the chatbot
bash
python script5.py

Stockbot loads the model from saved_model1/ and answers questions about a commodity's expected price direction.

Example inputs
help
list
will gold go up on 2026-03-10?
predict silver 2026-03-10
quit

Use the commodity names shown by the bot, and write dates as YYYY-MM-DD.

Troubleshooting
Problem	Fix
Model not found	Run python script4.py first and make sure saved_model1/ is in the same folder as script5.py.
Dataset download fails	Check your internet connection and that datasets is installed.
Unknown commodity	Use a name that matches the dataset's symbols.
Ideas for improvement
Add more indicators (RSI, MACD, Bollinger Bands).
Try a sequence model (LSTM).
Build a web interf
