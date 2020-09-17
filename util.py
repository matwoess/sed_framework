# -*- coding: utf-8 -*-
import os
import zipfile
from typing import List


def get_scene_classes(scene: str) -> List[str]:
    classes = []
    if scene in ['indoor', 'all']:
        classes.extend([
            "(object) rustling",
            "(object) snapping",
            "cupboard",
            "cutlery",
            "dishes",
            "drawer",
            "glass jingling",
            "object impact",
            "people walking",
            "washing dishes",
            "water tap running"])
    if scene in ['outdoor', 'all']:
        classes.extend([
            "(object) banging",
            "bird singing",
            "car passing by",
            "children shouting",
            "people speaking",
            "people walking",
            "wind blowing"])
    # add a class for "no event"
    classes.append('background')
    return classes


def flatten_dict(dictionary: dict, root: str = '', separator: str = '/') -> dict:
    def flatten(obj, string=''):
        if type(obj) == dict:
            string = string + separator if string else string
            for k in obj.keys():
                yield from flatten(obj[k], string + str(k))
        else:
            yield string, obj

    flat_dict = {k: v for k, v in flatten(dictionary, root)}
    return flat_dict


def zip_folder(folder: str, zipfile_name: str) -> None:
    archive = zipfile.ZipFile(f'{zipfile_name}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(folder):
        for file in files:
            archive.write(os.path.join(root, file))
    archive.close()


if __name__ == '__main__':
    zip_folder('results', 'results')
