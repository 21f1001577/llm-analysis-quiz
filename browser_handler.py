import logging
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logger = logging.getLogger(__name__)

class BrowserHandler:
    """Handle browser automation using Selenium"""

    def __init__(self, config):
        self.config = config
        self.driver = None
        self._init_driver()

    def _init_driver(self):
        """Initialize Chrome WebDriver for Railway"""
        try:
            chrome_options = Options()

            # Headless mode
            chrome_options.add_argument("--headless=new")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")

            # Use system-installed Chromium + chromedriver
            chrome_bin = os.getenv("CHROME_BIN", "/usr/bin/chromium")
            chrome_driver = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

            chrome_options.binary_location = chrome_bin

            service = Service(chrome_driver)
            self.driver = webdriver.Chrome(service=service, options=chrome_options)

            self.driver.set_page_load_timeout(self.config.BROWSER_TIMEOUT)
            logger.info("Browser initialized successfully on Railway")

        except Exception as e:
            logger.error(f"Failed to initialize browser: {str(e)}")
            raise

    def get_page_content(self, url):
        """Returns both text and html"""
        if not self.driver:
            raise RuntimeError("Browser driver not initialized")

        try:
            logger.info(f"Loading page: {url}")
            self.driver.get(url)

            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            time.sleep(1.2)

            text = self.driver.find_element(By.TAG_NAME, "body").text
            html = self.driver.page_source

            return {
                "text": text,
                "html": html
            }

        except Exception as e:
            logger.error(f"Error getting page content: {str(e)}")
            return None

    def close(self):
        """Close browser"""
        try:
            if self.driver:
                self.driver.quit()
                logger.info("Browser closed")
        except Exception as e:
            logger.error(f"Error closing browser: {str(e)}")
