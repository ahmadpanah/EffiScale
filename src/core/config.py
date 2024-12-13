# config.py
from typing import Dict
import yaml

class Config:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)