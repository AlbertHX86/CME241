import pandas as pd

INPUT_CSV = "S&P 500 Historical Data (1).csv"
OUTPUT_CSV = "sp500_market_state_2014_2023.csv"


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df = df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

    # Keep 2013 data to ensure MA200 can be built for 2014 onward.
    hist = df[(df["Date"] >= "2013-01-01") & (df["Date"] <= "2023-12-31")].copy()

    hist["ma50"] = hist["Price"].rolling(window=50, min_periods=50).mean()
    hist["ma200"] = hist["Price"].rolling(window=200, min_periods=200).mean()
    hist["bool"] = (hist["ma50"] > hist["ma200"]).astype(int)

    out = hist[(hist["Date"] >= "2014-01-01") & (hist["Date"] <= "2023-12-31")][["Date", "bool"]].copy()
    out = out.rename(columns={"Date": "datetime"})
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d")

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(out)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
