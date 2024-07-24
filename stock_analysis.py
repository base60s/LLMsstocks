import os
from langchain_groq import ChatGroq
import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from datetime import date
import matplotlib.pyplot as plt

# Function to get API key from user
def get_api_key():
    api_key = input("Please enter your Groq API key: ")
    return api_key

@tool
def get_stock_info(symbol, key):
    '''Return the correct stock info value given the appropriate symbol and key. Infer valid key from the user prompt; it must be one of the following:

    address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio
    
    If asked generically for 'stock price', use currentPrice
    '''
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]

@tool
def get_historical_price(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """
    data = yf.Ticker(symbol)
    hist = data.history(start=start_date, end=end_date)
    hist = hist.reset_index()
    hist[symbol] = hist['Close']
    return hist[['Date', symbol]]

def plot_price_over_time(historical_price_dfs):
    full_df = pd.DataFrame(columns=['Date'])
    for df in historical_price_dfs:
        full_df = full_df.merge(df, on='Date', how='outer')

    plt.figure(figsize=(12, 6))
    for column in full_df.columns[1:]:
        plt.plot(full_df['Date'], full_df[column], label=column)

    plt.title('Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]))
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.legend(title='Stock Symbol')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def call_functions(llm_with_tools, user_prompt):
    system_prompt = 'You are a helpful finance assistant that analyzes stocks and stock prices. Today is {today}'.format(today=date.today())
    
    messages = [SystemMessage(system_prompt), HumanMessage(user_prompt)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    historical_price_dfs = []
    symbols = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"get_stock_info": get_stock_info, "get_historical_price": get_historical_price}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        if tool_call['name'] == 'get_historical_price':
            historical_price_dfs.append(tool_output)
            symbols.append(tool_output.columns[1])
        else:
            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
    if len(historical_price_dfs) > 0:
        plot_price_over_time(historical_price_dfs)
    
        symbols = ' and '.join(symbols)
        messages.append(ToolMessage(f'A historical stock price chart for {symbols} has been generated.', tool_call_id=0))

    return llm_with_tools.invoke(messages).content

def main():
    api_key = get_api_key()
    os.environ['GROQ_API_KEY'] = api_key

    try:
        llm = ChatGroq(groq_api_key=api_key, model='llama3-8b-8192') #indicate the model you want to run
        
        tools = [get_stock_info, get_historical_price]
        llm_with_tools = llm.bind_tools(tools)

        print("Welcome to the Stock Market Analysis Tool powered by Groq and Llama 3")
        print("Try asking questions like:")
        print("- What is the current price of Meta stock?")
        print("- Show me the historical prices of Apple vs Microsoft stock over the past 6 months.")

        while True:
            user_question = input("\nAsk a question about a stock or multiple stocks (or type 'quit' to exit): ")
            
            if user_question.lower() == 'quit':
                break

            response = call_functions(llm_with_tools, user_question)
            print("\nResponse:", response)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please make sure your API key is correct and try again.")

if __name__ == "__main__":
    main()