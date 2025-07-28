import numpy as np 
import yfinance as yf 
import matplotlib.pyplot as plt

ticker = yf.Ticker("AAPL")
latest = ticker.history(period="1mo", interval="1d")

print(ticker)

latest_close = latest['Close'].iloc[-1]

S0 = latest_close
mu = 0.10
sigma = 0.1123
T=1
N=252
dt = T/N

Z = np.random.normal(0,1,N)
S = np.zeros(N)
S[0] = S0

for t in range(1,N):
    S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[t])

plt.plot(S)
plt.title("Stock Price through GBM")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()