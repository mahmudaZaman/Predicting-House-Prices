import yaml


class Config:
    _config = None

    @classmethod
    def load_config(cls, file_path='config.yaml'):
        if not cls._config:
            try:
                with open(file_path, 'r') as file:
                    cls._config = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Error: Config file '{file_path}' not found.")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file: {e}")
        return cls._config
