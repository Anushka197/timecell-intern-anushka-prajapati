import requests
import yfinance as yf
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate


# ---------------------- LOGGING SETUP ---------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("market_fetch.log"),
        logging.StreamHandler()
    ]
)


# ---------------------- DATA MODEL ---------------------- #
@dataclass
class AssetInfo:
    name: str
    price: Optional[float]
    currency: str


# ---------------------- UTILS ---------------------- #
def safe_request(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
    raise Exception("API request failed after retries")


def format_price(price: Optional[float]) -> str:
    return f"{price:,.2f}" if price is not None else "N/A"


def get_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


# ---------------------- FETCH FUNCTIONS ---------------------- #
def fetch_crypto(coin_id: str) -> AssetInfo:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    
    try:
        response = safe_request(url)
        data = response.json()
        
        price = float(data[coin_id]["usd"])
        logging.info(f"Fetched crypto: {coin_id}")
        
        return AssetInfo(name=coin_id.upper(), price=price, currency="USD")
    
    except Exception as e:
        logging.error(f"Crypto fetch failed ({coin_id}): {e}")
        return AssetInfo(name=coin_id.upper(), price=None, currency="USD")


def fetch_stock(ticker: str) -> AssetInfo:
    try:
        asset = yf.Ticker(ticker)
        hist = asset.history(period="1d")
        
        if hist.empty:
            raise ValueError("No data returned")
        
        price = float(hist["Close"].iloc[-1])
        
        # Try getting currency dynamically
        try:
            currency = asset.info.get("currency", "N/A")
        except:
            currency = "N/A"
        
        logging.info(f"Fetched stock: {ticker}")
        
        return AssetInfo(name=ticker.upper(), price=price, currency=currency)
    
    except Exception as e:
        logging.error(f"Stock fetch failed ({ticker}): {e}")
        return AssetInfo(name=ticker.upper(), price=None, currency="N/A")


# ---------------------- USER INPUT ---------------------- #
def get_user_assets() -> List[Tuple[str, str]]:
    print("Enter assets (type: crypto/stock). Type 'done' to finish.\n")
    
    assets = []
    
    while True:
        name = input("Asset (e.g., bitcoin, RELIANCE.NS): ").strip()
        
        if name.lower() == "done":
            break
        
        asset_type = input("Type (crypto/stock): ").strip().lower()
        
        if asset_type not in ("crypto", "stock"):
            print("Invalid type. Please enter 'crypto' or 'stock'.\n")
            continue
        
        assets.append((name, asset_type))
    
    return assets


# ---------------------- PARALLEL FETCH ---------------------- #
def fetch_all_assets(user_assets: List[Tuple[str, str]]) -> List[AssetInfo]:
    tasks = []
    
    for name, asset_type in user_assets:
        if asset_type == "crypto":
            tasks.append(lambda n=name: fetch_crypto(n))
        else:
            tasks.append(lambda n=name: fetch_stock(n))
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda f: f(), tasks))
    
    return results


# ---------------------- RENDER TABLE ---------------------- #
def render_table(assets: List[AssetInfo]):
    timestamp = get_timestamp()
    
    table = [
        [a.name, format_price(a.price), a.currency]
        for a in assets
    ]
    
    print(f"\nAsset Prices — fetched at {timestamp}\n")
    print(tabulate(table, headers=["Asset", "Price", "Currency"], tablefmt="grid"))
    print()


# ---------------------- MAIN ---------------------- #
def main():
    print("Fetching live market data...\n")
    
    user_assets = get_user_assets()
    
    if not user_assets:
        print("No assets entered. Exiting.")
        return
    
    results = fetch_all_assets(user_assets)
    
    # Warn for failed assets
    for asset in results:
        if asset.price is None:
            print(f"Warning: Failed to fetch {asset.name}")
    
    render_table(results)


if __name__ == "__main__":
    main()