import yfinance as yf
import requests
from datetime import datetime
import logging
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.ERROR,
    format="[ERROR] %(message)s"
)

@dataclass
class AssetInfo:
    name: str
    price: Optional[float]
    currency: str

def fetch_crypto_price_coingecko(coin_id: str, display_name: str, currency: str = "usd") -> AssetInfo:
    """Fetches real-time cryptocurrency data using the free CoinGecko API."""
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies={currency}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if coin_id in data and currency in data[coin_id]:
            price = float(data[coin_id][currency])
            return AssetInfo(name=display_name, price=price, currency=currency.upper())
        else:
            raise ValueError("Unexpected API response structure.")
            
    except Exception as e:
        logging.error(f"Failed to fetch {display_name} from CoinGecko: {e}")
        return AssetInfo(name=display_name, price=None, currency=currency.upper())

def fetch_market_price_yfinance(ticker: str, display_name: str, currency: str) -> AssetInfo:
    """Fetches market data (stocks/indices) using the yfinance library."""
    try:
        asset = yf.Ticker(ticker)
        todays_data = asset.history(period="1d")
        
        if not todays_data.empty:
            price = float(todays_data['Close'].iloc[-1])
            return AssetInfo(name=display_name, price=price, currency=currency.upper())
        else:
            raise ValueError("No price data returned from Yahoo Finance.")
            
    except Exception as e:
        logging.error(f"Failed to fetch {display_name} ({ticker}) from Yahoo Finance: {e}")
        return AssetInfo(name=display_name, price=None, currency=currency.upper())

def format_price(price: Optional[float]) -> str:
    """Formats the price with commas, or returns 'ERROR' if price is missing."""
    if price is None:
        return "ERROR"
    return f"{price:,.2f}"

def render_terminal_table(assets: list[AssetInfo]):
    """Renders a clean ASCII table to the terminal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    
    print(f"\nAsset Prices — fetched at {timestamp}")
    print("┌──────────┬───────────────┬──────────┐")
    print("│ Asset    │ Price         │ Currency │")
    print("├──────────┼───────────────┼──────────┤")
    
    for asset in assets:
        price_str = format_price(asset.price)
        # Pad strings to maintain table alignment
        print(f"│ {asset.name:<8} │ {price_str:>13} │ {asset.currency:<8} │")
        
    print("└──────────┴───────────────┴──────────┘\n")

def main():
    print("Fetching live market data. Please wait...")
    
    btc_data = fetch_crypto_price_coingecko(
        coin_id="bitcoin", 
        display_name="BTC", 
        currency="usd"
    )
    
    nifty_data = fetch_market_price_yfinance(
        ticker="^NSEI", 
        display_name="NIFTY", 
        currency="INR"
    )
    
    reliance_data = fetch_market_price_yfinance(
        ticker="RELIANCE.NS", 
        display_name="RELIANCE", 
        currency="INR"
    )
    
    portfolio_data = [btc_data, nifty_data, reliance_data]
    render_terminal_table(portfolio_data)

if __name__ == "__main__":
    main()