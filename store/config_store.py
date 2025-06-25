import os
from dotenv import load_dotenv

class ConfigStore:
    def __init__(self, env_path: str = '.env'):
        load_dotenv(env_path)

    def get(self, key: str, default=None):
        return os.getenv(key, default)