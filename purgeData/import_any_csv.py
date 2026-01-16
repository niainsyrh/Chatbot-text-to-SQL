# import_any_csv.py
import sys
import pandas as pd
from sqlalchemy import create_engine
from global_vars import DATABASE_URL  # uses your .env config

def auto_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to convert any column that looks like a date/time.
    Works for things like: 14/3/2001 1:30:55 PM, 2024-02-01, 3/5/24, etc.
    """
    for col in df.columns:
        col_lower = col.lower()
        if any(key in col_lower for key in ["date", "time", "datetime", "timestamp"]):
            # Try parsing as day-first (works for 14/3/2001 1:30:55 PM)
            try:
                df[col] = pd.to_datetime(df[col], errors="ignore", dayfirst=True)
            except Exception:
                # If it fails, just leave the column as-is
                pass
    return df

def import_csv_to_mysql(csv_path: str, table_name: str):
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # 1) Auto-parse date columns
    df = auto_parse_dates(df)

    # 2) Write to MySQL
    engine = create_engine(DATABASE_URL)
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    print(f"Imported into table `{table_name}`")

if __name__ == "__main__":
    # Example usage:
    #   python import_any_csv.py purgedata.csv purgedata
    #   python import_any_csv.py incidents.csv incidents
    if len(sys.argv) != 3:
        print("Usage: python import_any_csv.py <csv_path> <table_name>")
        sys.exit(1)

    csv_path = sys.argv[1]
    table_name = sys.argv[2]
    import_csv_to_mysql(csv_path, table_name)
