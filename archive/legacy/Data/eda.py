import pandas as pd
import matplotlib.pyplot as plt

#Loading datasets
cpi = pd.read_csv('cpi_data.csv', parse_dates=['Date'])
interest = pd.read_csv('interest_rate_data.csv', parse_dates=['Date'])
unemployment = pd.read_csv('unemployment_data.csv', parse_dates=['Date'])

#Merging datasets
df = cpi.merge(interest, on='Date', how='outer') \
        .merge(unemployment, on='Date', how='outer')

#Filtering between 1980 and 2025
df = df[(df['Date'].dt.year >= 1980) & (df['Date'].dt.year <= 2025)]
df.sort_values(by='Date', inplace=True)

#Calculating CPI YoY growth (inflation)
df['CPI_growth'] = df['CPI'].pct_change(periods=12) * 100  # Assumes monthly data


#Plot all on one y-axis
plt.figure(figsize=(15,8))

plt.plot(df['Date'], df['CPI_growth'], label='Inflation Rate (CPI YoY %)', color='blue')

#Plot raw unemployment rate (percentage level)
plt.plot(df['Date'], df['UnemploymentRate'], label='Unemployment Rate (%)', color='green')

#Plot Fed Funds Rate (percentage level)
plt.plot(df['Date'], df['FedFundsRate'], label='Fed Funds Rate (%)', color='red')

plt.ylabel('Percentage (%)')
plt.xlabel('Year')
plt.title('Inflation, Unemployment, and Interest Rates (1980–2025)')
plt.legend()
plt.tight_layout()
plt.show()


