import yaml


def read_file(yml_file_path) -> dict:
    with open(yml_file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def save_file(data, yml_file_path):
    with open(yml_file_path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)
