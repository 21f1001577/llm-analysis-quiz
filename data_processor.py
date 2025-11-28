import logging
import os
import requests
import pandas as pd
import json
from pathlib import Path
import PyPDF2
from bs4 import BeautifulSoup
import base64

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data downloading and processing"""
    
    def __init__(self, config):
        self.config = config
        self.download_folder = Path(config.DOWNLOAD_FOLDER)
        self.download_folder.mkdir(exist_ok=True)
    
    def download_file(self, url):
        try:
            r = requests.head(url, allow_redirects=True, timeout=10)
            ctype = r.headers.get('Content-Type','').lower()
        except Exception:
            ctype = ''
        # If it's HTML, we may want browser-rendered page instead of downloading
        if 'text/html' in ctype or url.endswith('/'):
            # Let caller render the HTML; return a marker or save raw content
            # Optionally download the HTML
            r = requests.get(url, timeout=15)
            path = os.path.join(self.config.DOWNLOAD_FOLDER, os.path.basename(url) or 'downloaded_page.html')
            with open(path, 'wb') as f:
                f.write(r.content)
            return path
        # CSV
        if 'text/csv' in ctype or url.lower().endswith('.csv'):
            r = requests.get(url, timeout=30)
            path = os.path.join(self.config.DOWNLOAD_FOLDER, os.path.basename(url))
            with open(path, 'wb') as f:
                f.write(r.content)
            return path
        # PDF
        if 'pdf' in ctype or url.lower().endswith('.pdf'):
            r = requests.get(url, timeout=30)
            path = os.path.join(self.config.DOWNLOAD_FOLDER, os.path.basename(url))
            with open(path, 'wb') as f:
                f.write(r.content)
            return path
        # other: try to download
        r = requests.get(url, timeout=30)
        path = os.path.join(self.config.DOWNLOAD_FOLDER, os.path.basename(url) or 'download')
        with open(path, 'wb') as f:
            f.write(r.content)
        return path

    
    def process_file(self, filepath):
        """Process a file based on its type"""
        try:
            filepath = Path(filepath)
            extension = filepath.suffix.lower()
            
            logger.info(f'Processing file: {filepath} (type: {extension})')
            
            if extension == '.pdf':
                return self.process_pdf(filepath)
            elif extension in ['.csv']:
                return self.process_csv(filepath)
            elif extension in ['.xlsx', '.xls']:
                return self.process_excel(filepath)
            elif extension in ['.json']:
                return self.process_json(filepath)
            elif extension in ['.txt']:
                return self.process_text(filepath)
            elif extension in ['.html', '.htm']:
                return self.process_html(filepath)
            else:
                logger.warning(f'Unsupported file type: {extension}')
                return None
                
        except Exception as e:
            logger.error(f'Error processing file: {str(e)}')
            return None
    
    def process_pdf(self, filepath):
        """Extract text and tables from PDF"""
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                
                data = {
                    'pages': len(reader.pages),
                    'text_by_page': {}
                }
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    data['text_by_page'][i + 1] = text
                
                logger.info(f'Extracted {len(reader.pages)} pages from PDF')
                return data
                
        except Exception as e:
            logger.error(f'Error processing PDF: {str(e)}')
            return None
    
    def process_csv(self, filepath):
        """Process CSV file"""
        try:
            df = pd.read_csv(filepath)
            
            data = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'data': df.to_dict('records'),
                'summary': df.describe().to_dict()
            }
            
            logger.info(f'Processed CSV: {df.shape[0]} rows, {df.shape[1]} columns')
            return data
            
        except Exception as e:
            logger.error(f'Error processing CSV: {str(e)}')
            return None
    
    def process_excel(self, filepath):
        """Process Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(filepath)
            
            data = {
                'sheets': excel_file.sheet_names,
                'data': {}
            }
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet_name)
                data['data'][sheet_name] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'data': df.to_dict('records'),
                    'summary': df.describe().to_dict()
                }
            
            logger.info(f'Processed Excel with {len(excel_file.sheet_names)} sheets')
            return data
            
        except Exception as e:
            logger.error(f'Error processing Excel: {str(e)}')
            return None
    
    def process_json(self, filepath):
        """Process JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info('Processed JSON file')
            return data
            
        except Exception as e:
            logger.error(f'Error processing JSON: {str(e)}')
            return None
    
    def process_text(self, filepath):
        """Process text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            data = {
                'content': content,
                'lines': content.split('\n'),
                'length': len(content)
            }
            
            logger.info(f'Processed text file: {len(content)} characters')
            return data
            
        except Exception as e:
            logger.error(f'Error processing text: {str(e)}')
            return None
    
    def process_html(self, filepath):
        """Process HTML file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                html = f.read()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            data = {
                'title': soup.title.string if soup.title else None,
                'text': soup.get_text(),
                'links': [a.get('href') for a in soup.find_all('a')]
            }
            
            logger.info('Processed HTML file')
            return data
            
        except Exception as e:
            logger.error(f'Error processing HTML: {str(e)}')
            return None
    
    def encode_file_to_base64(self, filepath):
        """Encode a file to base64 for submission"""
        try:
            with open(filepath, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
            
            return encoded
            
        except Exception as e:
            logger.error(f'Error encoding file: {str(e)}')
            return None