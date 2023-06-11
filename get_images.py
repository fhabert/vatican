from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
import time
import requests


def get_images():
    URL = "https://digi.vatlib.it/search?k_f=0&k_v=latin"
    download_directory = os.path.join(os.path.expanduser("~"), "Downloads")
    chrome_options = Options()
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"--download.default_directory={download_directory}")
    # chrome_options.add_argument("--headless")
    executable_p = "C:\Program Files\Google\Chrome\chromedriver.exe"
    driver = webdriver.Chrome(executable_path=executable_p, options=chrome_options)
    driver.get(URL)

    counter = 0

    for i in range(20):
        if i != 1:
            all_records = driver.find_elements(By.CLASS_NAME, "row-search-result-record")
            div_container = all_records[i].find_element(By.CLASS_NAME, "block-search-result-record-body")
            link = div_container.find_element(By.CLASS_NAME, "box-search-result-view-link")
            link.click()

            time.sleep(2)
            pic_list = driver.find_element(By.CLASS_NAME, "aiv-scenelist-list")
            items = pic_list.find_elements(By.XPATH, ".//a")

        #puis looper sur les photos

            for j in range(15,20):
                # link = items[i].find_element(By.CLASS_NAME, "aiv-gvf-thumbnail")
                # link = item.find_element_by_xpath(".//a[@class='aiv-gvf-thumbnail'")
                driver.execute_script("arguments[0].click();", items[j])
                time.sleep(1)
                # wait = WebDriverWait(driver, 10)
                # element = wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "btn-jpeg-panel")))
                btn_down = driver.find_element_by_xpath("//a[@title='Download JPEG']")
                btn_down.click()
                print(btn_down.get_attribute("title"))
                form = driver.find_element(By.CLASS_NAME, "viewer-jpeg-download-form")
                form_select = form.find_elements(By.XPATH, ".//label")
                form_select[1].click()
                # driver.execute_script("arguments[0].click();", btn_down)
                time.sleep(1)
                download = driver.find_element(By.CLASS_NAME, "btn-download-jpeg")
                time.sleep(1)
                download.click()
                download_path = "D:/vatican/hq_database"
                href = download.get_attribute("href")
                file_name = download.get_attribute("download")
                extension = os.path.splitext(file_name)[1]
                unique_file = "image" + str(counter) + extension
                file_path = os.path.join(download_path, unique_file)
                with open(file_path, "wb") as f:
                    f.write(requests.get(href).content)
                counter += 1
                btn_down.click()
                time.sleep(1)
            driver.back()

    driver.quit()


get_images()