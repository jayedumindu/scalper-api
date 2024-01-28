import aiohttp
import asyncio
import pandas as pd
import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import pytz
from binance.client import Client
import time
import traceback
import csv

api_key='18abadde0f5683e7334844c452bdcbdcd516b76bef5d9e308b5226cb25386056'
api_secret='96208bcf396a55548364510de2363b6e0aa18a8ab052ec013761012b954cffa4'

# Initialize the Binance client
# Use testnet=True for testing
binance_futures = Client(api_key=api_key, api_secret=api_secret, testnet=False)

# Define a mapping of intervals to their respective resampling rules
interval_resampling_map = {
    '30s': '30S',
    '1m': '1T',
    '3m': '3T',
    '5m': '5T',
    '15m': '15T',
    '30m': '30T',
    '1h': '1H',
    '2h': '2H',
    '4h': '4H',
    '6h': '6H',
    '8h': '8H',
    '12h': '12H',
    '1d': '1D',
    '3d': '3D',
    '1w': '1W',
    '1M': '1M'
}

ist_timezone = pytz.timezone('Asia/Kolkata')

logged_candles = set()  # Set to store the timestamps of logged candles

macd_crossover_signal = 0
macd_turning_point_signal = 0
rsi_signal = 0
buy_sell_pressure_signal = 0
shooting_star_signal = 0
moving_average_trend_signal = 0
last_candle_signal = 0

# Define the UTC offset for the 'Asia/Kolkata' timezone
# IST (Indian Standard Time) is UTC+5:30
utc_offset = timedelta(hours=5, minutes=30)


def is_candle_closed(timestamp, interval_minutes):

    # Convert current time to UTC
    current_time = datetime.now(timezone.utc)

    # Adjust the timestamp to the 'Asia/Kolkata' timezone
    adjusted_timestamp = timestamp.replace(tzinfo=timezone(utc_offset))

    # Calculate the expected closing time based on the interval
    expected_closing_time = adjusted_timestamp + \
        timedelta(minutes=interval_minutes)

    # Compare the current time with the expected closing time
    return current_time >= expected_closing_time

def get_latest_closed_candle(candlestick_data, closing_duration=5):
    latest_candle_index = candlestick_data.index[-1]

    # Check if the latest candle is closed
    if is_candle_closed(latest_candle_index, closing_duration):
        # Return the entire candlestick data of the latest closed candle
        return candlestick_data.loc[latest_candle_index]

    # If the latest candle is not closed, search for the previous last closed candle
    for i in range(len(candlestick_data) - 2, -1, -1):
        if is_candle_closed(candlestick_data.index[i], closing_duration):
            # Return the entire candlestick data of the last closed candle
            return candlestick_data.loc[candlestick_data.index[i]]
    print("returning none")
    return None  # If no closed candle is found, return None

def detect_last_candle_signal(candlestick_data):
    global last_candle_signal

    # Get the latest candlestick from the provided data
    latest_candle = get_latest_closed_candle(candlestick_data)

    if latest_candle is not None:
        open_price = latest_candle['Open']
        close_price = latest_candle['Close']
        high_price = latest_candle['High']
        low_price = latest_candle['Low']

        body_size = abs(open_price - close_price)
        total_range = high_price - low_price

        # Extract the timestamp of the latest candle
        timestamp = latest_candle.name

        # Log the timestamp
        print(f"Timestamp of the latest candle: {timestamp}")

        # Check if the latest candle exhibits a specific pattern
        if body_size > 0.5 * total_range:
            if close_price > open_price and (high_price - close_price) <= 0.2 * body_size:
                last_candle_signal = 1  # Bullish signal
                print("bullish candle")
            elif close_price < open_price and (close_price - low_price) <= 0.2 * body_size:
                last_candle_signal = 0  # Bearish signal
                print("bearish candle")
            else:
                last_candle_signal = None  # No significant signal
                print("not significant candle")

        else:
            last_candle_signal = None  # No significant signal

        return timestamp

def detect_macd_turns(macd_line):
    buy_signals = []
    sell_signals = []
    above_zero = macd_line > 0

    for i in range(1, len(macd_line)):
        if above_zero.iloc[i] and not above_zero.iloc[i - 1]:
            buy_signals.append(i)
        elif not above_zero.iloc[i] and above_zero.iloc[i - 1]:
            sell_signals.append(i)

    return buy_signals, sell_signals

def detect_macd_crossovers(macd_line, signal_line):
    buy_signals = []
    sell_signals = []
    crossover_above_zero = (macd_line > signal_line) & (macd_line > 0)

    for i in range(1, len(macd_line)):
        if crossover_above_zero.iloc[i] and not crossover_above_zero.iloc[i - 1]:
            buy_signals.append(i)
        elif not crossover_above_zero.iloc[i] and crossover_above_zero.iloc[i - 1]:
            sell_signals.append(i)

    return buy_signals, sell_signals

async def fetch_binance_klines(symbol, interval, contract, limit=1000):
    base_url = 'https://fapi.binance.com/fapi/v1/continuousKlines'

    # Define the query parameters
    params = {
        'pair': symbol,
        'interval': interval,
        'limit': limit,
        'contractType': contract
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(base_url, params=params) as response:
                response.raise_for_status()
                kline_data = await response.json()
                return kline_data
        except aiohttp.ClientError as e:
            print(f"An error occurred: {e}")
            return None

async def fetch_binance_price(symbol):
    price_url = f'https://api.binance.com/api/v3/ticker/price'

    # Define the query parameters
    params = {
        'symbol': symbol
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(price_url, params=params) as response:
                response.raise_for_status()
                price_data = await response.json()
                return float(price_data['price'])
        except aiohttp.ClientError as e:
            print(f"An error occurred while fetching price: {e}")
            return None

def calculate_rsi(candlestick_data):
    if len(candlestick_data) < 14:
        print("Not enough data to calculate RSI.")
        return None

    # Extract the closing prices from the candlestick_data DataFrame
    closes = candlestick_data['Close']

    # Calculate price differences and classify them as gains or losses
    price_diff = closes.diff()
    gains = np.where(price_diff > 0, price_diff, 0)
    losses = np.where(price_diff < 0, -price_diff, 0)

    avg_gain = gains[-14:].mean()
    avg_loss = losses[-14:].mean()

    if avg_loss == 0:
        rsi = 100.0
    else:
        relative_strength = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + relative_strength))

    return rsi

def calculate_macd(df, short_window, long_window, signal_window=9):
    # Calculate the short-term and long-term exponential moving averages (EMAs)
    ema_short = df['Close'].ewm(
        span=short_window, min_periods=short_window, adjust=False).mean()
    ema_long = df['Close'].ewm(
        span=long_window, min_periods=long_window, adjust=False).mean()

    # Calculate the MACD line
    macd_line = ema_short - ema_long

    # Calculate the signal line (9-day EMA of MACD)
    signal_line = macd_line.ewm(
        span=signal_window, min_periods=signal_window, adjust=False).mean()

    return macd_line, signal_line

# for determining long term moving average using 200 MA with the last 20 closing prices
def detect_trend(df_resampled, window):
    global moving_average_trend_signal
    if len(df_resampled) >= 100:
        # Calculate the moving average
        ma_200 = df_resampled['Close'].rolling(window).mean()

        # Get the last 20 closing prices
        last_20_closes = df_resampled['Close'][-20:]

        # Check if all of the last 20 closing prices are above the MA
        if all(last_20_closes > ma_200.iloc[-20:]):
            moving_average_trend_signal = 1
            return f"MA {window} says : Uptrend"
        # Check if all of the last 20 closing prices are below the MA
        elif all(last_20_closes < ma_200.iloc[-20:]):
            moving_average_trend_signal = 0
            return f"MA {window} says : Downtrend"

    # If conditions are not met, return "No Trend" or None
    return "No Trend"

def detect_buy_sell_pressure(candlestick_data):
    global logged_candles
    global buy_sell_pressure_signal
    global shooting_star_signal

    # Filter out rows where Close is not null (i.e., closed candles)
    closed_candles = candlestick_data[candlestick_data['Close'].notnull()]

    # Get the last two closed candles
    last_two_closed_candles = closed_candles.iloc[-2:]
    last_five_closed_candles = closed_candles.iloc[-5:]

    # Check if there are exactly two closed candles
    if len(last_two_closed_candles) == 2:
        # Check if both last candles are strong bearish
        if is_strong_bearish(last_two_closed_candles.iloc[0]) and is_strong_bearish(last_two_closed_candles.iloc[1]):
            print("Strong bearish pattern detected. Close the position.")
            buy_sell_pressure_signal = 0  # Set to 0 for sell signal

        # Check if both last candles are strong bullish
        if is_strong_bullish(last_two_closed_candles.iloc[0]) and is_strong_bullish(last_two_closed_candles.iloc[1]):
            print("Strong bullish pattern detected. Close the position.")
            buy_sell_pressure_signal = 1  # Set to 1 for buy signal

    # Analyze the last 5 candles for shooting star patterns
    for _, candle in last_five_closed_candles.iterrows():
        is_shooting, direction = is_shooting_star(candle)
        if is_shooting:
            shooting_star_signal = direction

# for identifying sudden bull/bear signals
def is_shooting_star(candle):
    # Calculate body size and total range
    body_size = abs(candle['Open'] - candle['Close'])
    total_range = candle['High'] - candle['Low']

    # Define the criteria for a shooting star pattern
    is_shooting_star_pattern = body_size < 0.2 * total_range

    bullish_shooting_star = is_shooting_star_pattern and (
        candle['High'] - candle['Open'] >= 0.7 * body_size)
    bearish_shooting_star = is_shooting_star_pattern and (
        candle['Open'] - candle['Low'] >= 0.7 * body_size)

    if bullish_shooting_star:
        return True, 1  # Bullish shooting star
    elif bearish_shooting_star:
        return True, 0  # Bearish shooting star

    return False, None

def is_strong_bearish(candle):
    # Define your criteria for a strong bearish candle
    body_size = abs(candle['Open'] - candle['Close'])
    total_range = candle['High'] - candle['Low']
    return body_size > 0.8 * total_range and candle['Close'] < candle['Open']

def is_strong_bullish(candle):
    # Define your criteria for a strong bullish candle
    body_size = abs(candle['Open'] - candle['Close'])
    total_range = candle['High'] - candle['Low']
    return body_size > 0.8 * total_range and candle['Close'] > candle['Open']

async def generate_realtime_candlestick_data(symbol, interval, contract, limit=200):

    # signals
    global macd_crossover_signal
    global macd_turning_point_signal
    global rsi_signal
    global buy_sell_pressure_signal
    global moving_average_trend_signal
    global last_candle_signal
    global shooting_star_signal
   
    try:

        kline_data = await fetch_binance_klines(symbol, interval, contract, limit)

        if kline_data:

            # Extract OHLCV data and create a DataFrame
            ohlcv_data = [[
                datetime.fromtimestamp(
                    entry[0] / 1000, tz=pytz.utc).astimezone(ist_timezone),
                float(entry[1]),  # Open
                float(entry[2]),  # High
                float(entry[3]),  # Low
                float(entry[4]),  # Close
                float(entry[5])   # Volume
            ] for entry in kline_data]

            df = pd.DataFrame(ohlcv_data, columns=[
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)

            resampling_rule = interval_resampling_map.get(interval)

            if resampling_rule:

                df_resampled = df.resample(resampling_rule).asfreq()

                df_resampled = df_resampled.dropna()

                current_price = await fetch_binance_price(symbol)

                if current_price is not None:

                    # MA trend
                    detect_trend(df_resampled, 20)

                    # Buy/Sell pressure | Shooting Start
                    detect_buy_sell_pressure(df_resampled)

                    # Last Candle
                    detect_last_candle_signal(df_resampled)

                    # RSI
                    if len(df_resampled) >= 14:
                        last_14_candlesticks = df_resampled[-14:]
                        rsi_value = calculate_rsi(last_14_candlesticks)
                        if rsi_value >= 55:
                            rsi_signal = 1
                        elif rsi_value <= 45:
                            rsi_signal = 0
                        else:
                            rsi_signal = None
                    else:
                        print("Not enough data to calculate RSI.")


                    # Calculate MACD, signal line
                    short_window = 20
                    long_window = 50
                    macd_line, signal_line = calculate_macd(
                        df_resampled, short_window, long_window)

                    buy_turns, sell_turns = detect_macd_turns(macd_line)
                    buy_crossovers, sell_crossovers = detect_macd_crossovers(
                        macd_line, signal_line)

                    latest_buy_crossover = max(
                        buy_crossovers, default=None)
                    latest_sell_crossover = max(
                        sell_crossovers, default=None)

                    # MACD crossover 
                    if latest_buy_crossover is not None and (latest_sell_crossover is None or latest_buy_crossover > latest_sell_crossover):
                        latest_crossover_timestamp = latest_buy_crossover
                        macd_crossover_signal = 1  # Set to 1 for buy crossover
                    elif latest_sell_crossover is not None and (latest_buy_crossover is None or latest_sell_crossover > latest_buy_crossover):
                        latest_crossover_timestamp = latest_sell_crossover
                        macd_crossover_signal = 0  # Set to 0 for sell crossover
                    else:
                        latest_crossover_timestamp = None
                        macd_crossover_signal = 0  # Default to 0 if no crossovers are found

                    latest_buy_turn = max(buy_turns, default=None)
                    latest_sell_turn = max(sell_turns, default=None)

                    # MACD turning point 
                    if latest_buy_turn is not None and (latest_sell_turn is None or latest_buy_turn > latest_sell_turn):
                        macd_turning_point_signal = 1  # Set to 1 for buy turn
                    elif latest_sell_turn is not None and (latest_buy_turn is None or latest_sell_turn > latest_buy_turn):
                        macd_turning_point_signal = 0  # Set to 0 for sell turn
                    else:
                        macd_turning_point_signal = 0  # Default to 0 if no turns are found

                    # return a dictianory with all the values 
                    signals = {
                        'MACD Crossover Signal': macd_crossover_signal,
                        'Last Candle Signal': last_candle_signal,
                        'Shooting Star Signal': shooting_star_signal,
                        'Moving Average Trend Signal': moving_average_trend_signal,
                        'MACD Turning Point Signal': macd_turning_point_signal,
                        'RSI Signal': rsi_signal,
                        'Buy/Sell Pressure Signal': buy_sell_pressure_signal,  
                    }

                    return signals

                else:
                    print("No data to plot.")
            else:
                print(f"Unsupported interval: {interval}")

    except Exception as e:
        print(f"Exception occurred: {e}")
        traceback.print_exc() 
        with open("error_log.txt", "a") as f:
            f.write(f"Exception occurred: {e}\n{traceback.format_exc()}\n")
        time.sleep(60)  




