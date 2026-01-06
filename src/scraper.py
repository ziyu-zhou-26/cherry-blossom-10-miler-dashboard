from selenium import webdriver
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from html import unescape
import pandas as pd
import time
import logging
from pathlib import Path

# src/
SRC_DIR = Path(__file__).resolve().parent

# project root/
PROJECT_ROOT = SRC_DIR.parent

# data/raw/
CSV_DIR = PROJECT_ROOT / "data" / "raw"
CSV_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='CherryBlossomScraper.log',
    filemode='a'  # Append to existing log file; use 'w' to overwrite
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.page_load_strategy='none'
chrome_path = ChromeDriverManager().install()
chrome_service = Service(chrome_path)
driver = Chrome(options=options, service=chrome_service)

url = 'https://www.timingproductions.com/results-site/cherry-blossom'
driver.get(url)

max_pages = 787
page = 1

allRows = []

wait = WebDriverWait(driver,10)

def checked_text(x):
    if not x or x == ', ':
        return ''
    return x

while page <= max_pages:
    logging.info(f'Scraping page {page}...')
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'tr.cbResultSetDataRow')))
    rows = driver.find_elements(By.CSS_SELECTOR, 'tr.cbResultSetDataRow')
    
    for i, row in enumerate(rows):
        row_html = row.get_attribute('innerHTML')
        if 'Disqualified' in row_html or 'Ineligible for Scoring' in row_html:
            logging.info(f'Skipping row {i}: Disqualified or Ineligible for Scoring')
            continue
        
        newRow = {}
        
        try:
            name = row.find_element(By.CSS_SELECTOR, "div[style*='font-weight:bold']").text
            rawinfo = row.find_element(By.CSS_SELECTOR, "div[style*='font-size: 16px']").text
            cleaninfo = unescape(rawinfo)
            parts = [p.strip() for p in cleaninfo.split('|')]
            gender_age = parts[0].split('-')
            race = parts[2]
            citystate_country = parts[3].rsplit(' ', 1)
            overallplace = row.find_element(By.XPATH, './td[2]').text.strip()
            genderplace = row.find_element(By.XPATH, './td[3]').text.strip()
            ageplace = row.find_element(By.XPATH, './td[4]').text.strip()
            finishtime = row.find_element(By.XPATH, './td[5]').text.strip()
            pace = row.find_element(By.XPATH, './td[6]').text.strip()
        
            newRow.update({
                'Name': checked_text(name),
                'Gender': checked_text(gender_age[0]),
                'Age': checked_text(gender_age[1]),
                'Race': checked_text(race),
                'State': checked_text(citystate_country[0][-2:]),
                'Country': checked_text(citystate_country[1]),
                'Overall Place': checked_text(overallplace),
                'Gender Place': checked_text(genderplace),
                'Age Group Place': checked_text(ageplace),
                'Finish Time': checked_text(finishtime),
                'Pace': checked_text(pace)
            })
            allRows.append(newRow)     
        except Exception as e:
            logging.warning(f'Skipping row {i} on page {page} due to error: {e}')
            continue
    if page % 20 == 0:
        pd.DataFrame(allRows).to_csv(
    CSV_DIR / f"CherryBlossom2025_partial_p{page}.csv",
    index=False
    )
        logging.info(f'Saved data at page {page}')

    try:
        next_button = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, '[Next >>]')))
        logging.info(f'Found [Next >>] button on page {page}')
        
        old_row = rows[0]
        next_button.click()
        logging.info(f'Clicked [Next >>] on page {page}')
        
        try:
            WebDriverWait(driver, 10).until(EC.staleness_of(old_row))
            logging.info(f'Page changed after clicking [Next >>] on page {page}')
            page += 1
        except Exception:
            logging.warning(f'Page did not change after clicking [Next >>] on page {page}')
            break
        
    except Exception as e:
        # Expected behavior on last page - no [Next >>] button exists
        logging.error(f'Stopped scraping. Error: {e}')
        break

driver.quit()

csvrows = pd.DataFrame(allRows)
csvrows.to_csv(
    CSV_DIR / "CherryBlossom2025_final.csv",
    index=False
)

logging.info(f'Total records scraped: {len(allRows)}')
logging.info('Scraping complete. Final CSV saved as CherryBlossom2025_final.csv')