from src.neuronTracer import *
from src.neuronTracer.entity.config import DataIngestionConfig
import os
import urllib.request as request
import zipfile
from pathlib import Path
from src.neuronTracer import logger
from src.neuronTracer.utils.common import get_size
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            local_file_path= self.config.local_data_file

            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # 2. Check if file already exists to save time/bandwidth
            if not os.path.exists(local_file_path):
                logger.info(f"Starting download from {dataset_url}...")
            
                # Use urllib for a clean, direct download
                filename, headers = request.urlretrieve(
                    url = dataset_url,
                    filename = local_file_path
                )
                logger.info(f"Download complete! File saved as: {filename}")
                logger.info(f"Info from server: \n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(local_file_path))}")

        except Exception as e:
            logger.error(f"Error occurred while downloading: {e}")
            raise e


    
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)