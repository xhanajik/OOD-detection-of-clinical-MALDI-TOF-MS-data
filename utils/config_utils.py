import yaml


def read_config(yaml_file):
    with open(yaml_file, "r") as file:
        file = yaml.safe_load(file)
    return file["main"], file["dataset"], file["ood"]#, file["training"]