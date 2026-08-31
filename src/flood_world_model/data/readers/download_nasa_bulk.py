import earthaccess
import os
import time
from datetime import datetime, timedelta

OUTPUT_DIR = "data/raw/nasa"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Login using .netrc (no interactive prompts)
earthaccess.login(strategy="netrc", persist=True)

bbox = (88.0, 20.5, 92.5, 26.5)
start_year = 2015
end_year = 2026

total_files = 0
total_skipped = 0

print("=" * 60)
print("🚀 Downloading NASA IMERG (2016–2023) using earthaccess")
print("   This will resume from where you left off.")
print("=" * 60)

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        month_str = f"{month:02d}"
        start_date = f"{year}-{month_str}-01"
        
        if month == 12:
            end_date = f"{year}-12-31"
        else:
            next_month = f"{year}-{month+1:02d}-01"
            end_date = (datetime.strptime(next_month, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"\n📅 Searching {year}-{month_str}...", end=" ")
        
        try:
            results = earthaccess.search_data(
                short_name="GPM_3IMERGDE",
                cloud_hosted=False,
                bounding_box=bbox,
                temporal=(start_date, end_date)
            )
        except Exception as e:
            print(f"❌ Search failed: {e}")
            time.sleep(10)
            continue
        
        if not results:
            print("⚠️ No files found.")
            continue
        
        # Check which files already exist
        files_to_download = []
        for granule in results:
            # Get the filename from the data link
            try:
                # earthaccess gives us the data links
                links = granule.data_links()
                if links:
                    filename = links[0].split('/')[-1]
                    local_path = os.path.join(OUTPUT_DIR, filename)
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000000:
                        total_skipped += 1
                    else:
                        files_to_download.append(granule)
            except Exception as e:
                print(f"⚠️ Could not get filename: {e}")
                files_to_download.append(granule)
        
        if files_to_download:
            print(f"Downloading {len(files_to_download)} files...")
            try:
                # ✅ REMOVED timeout argument
                earthaccess.download(files_to_download, OUTPUT_DIR)
                total_files += len(files_to_download)
                print(f"✅ Completed {year}-{month_str}")
            except Exception as e:
                print(f"❌ Download failed for {year}-{month_str}: {e}")
                # We'll retry in next run
        else:
            print(f"✅ All files already downloaded for {year}-{month_str}")

print("\n" + "=" * 60)
print(f"🎉 Summary:")
print(f"   New files downloaded: {total_files}")
print(f"   Files skipped (already exist): {total_skipped}")
print(f"📁 Files saved to: {OUTPUT_DIR}")
print("=" * 60)