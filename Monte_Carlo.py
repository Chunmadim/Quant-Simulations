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
M = 10000
price_paths = np.zeros((M,N+1))
price_paths[:,0] = S0


Z = np.random.normal(0,1,size=(M,N))

for t in range(1, N+1):
    price_paths[:,t] = price_paths[:,t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[:,t-1])

final_prices = price_paths[:,-1]

fig, axs = plt.subplots(1,2,figsize=(14,7))

for i in range(10000):
    axs[0].plot(price_paths[i])
axs[0].set_title("Price Paths")
axs[0].set_xlabel("Days")
axs[0].set_ylabel("Price")

axs[1].hist(final_prices,bins=100, edgecolor ="black")
axs[1].set_title("Distribution Simulated Prices")
axs[1].set_xlabel("Final Price")
axs[1].set_ylabel("Frequency")

mean_price = np.mean(final_prices)
var = np.percentile(final_prices, 5)
prob_loss = np.mean(final_prices < S0)


print(f"Mean Final Price: ${mean_price:.2f}\n, VaR: ${var:.2f} (loss: {S0-var:.2f})\n, Probability of Loss: {prob_loss:.2%}")


plt.show()