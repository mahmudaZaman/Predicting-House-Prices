import json

from pydantic import BaseModel, ConfigDict
import yaml
from pydantic import ValidationError


class FilesConfig(BaseModel):
    raw_data: str
    raw_data_pkl: str
    train_data: str
    test_data: str
    output_model_pkl: str


class StorageConfig(BaseModel):
    bucket_name: str
    files: FilesConfig


class AppConfig(BaseModel):
    storage: StorageConfig


# Load YAML content
with open('config.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)
# Parse YAML data using pydantic
try:
    app_config = AppConfig.model_validate(yaml_data)
    print(app_config.storage)

except ValidationError as e:
    print(f"Validation error: {e}")
