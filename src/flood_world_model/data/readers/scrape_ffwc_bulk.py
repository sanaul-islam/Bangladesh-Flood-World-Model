import requests
import os
from datetime import datetime, timedelta

OUTPUT_DIR = "data/raw/ffwc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Log missing dates
missing_log = f"{OUTPUT_DIR}/ffwc_missing_dates.txt"
missing_entries = []

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)

current = start_date
while current <= end_date:
    date_str = current.strftime("%Y-%m-%d")
    year = current.strftime("%Y")
    month = current.strftime("%m")
    day = current.strftime("%d")

    # Try Excel format first (most common)
    url = f"http://www.ffwc.gov.bd/data/water_level/station_wise_{year}_{month}_{day}.xls"
    
    try:
        response = requests.get(url, timeout=20)
        if response.status_code == 200:
            filename = f"{OUTPUT_DIR}/ffwc_{year}_{month}_{day}.xls"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✅ {date_str} downloaded.")
        else:
            # Try CSV fallback
            url_csv = f"http://www.ffwc.gov.bd/data/water_level/station_wise_{year}_{month}_{day}.csv"
            response_csv = requests.get(url_csv, timeout=20)
            if response_csv.status_code == 200:
                filename = f"{OUTPUT_DIR}/ffwc_{year}_{month}_{day}.csv"
                with open(filename, 'wb') as f:
                    f.write(response_csv.content)
                print(f"✅ {date_str} downloaded (CSV).")
            else:
                print(f"❌ {date_str} not found (404).")
                missing_entries.append(date_str)
    except Exception as e:
        print(f"⚠️ Error on {date_str}: {e}")
        missing_entries.append(date_str)

    current += timedelta(days=1)

# Save missing dates log
if missing_entries:
    with open(missing_log, 'w') as f:
        f.write("\n".join(missing_entries))
    print(f"📝 Missing dates logged to {missing_log}")

print("🎉 FFWC scraping complete!")