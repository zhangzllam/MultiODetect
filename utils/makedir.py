import os


def make_dir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
