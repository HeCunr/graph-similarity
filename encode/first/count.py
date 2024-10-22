import glob
import os

import ezdxf
import pandas as pd

def count_entity_types(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    entity_types = {}
    for entity in modelspace:

        entity_type = entity.dxftype()
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

    df = pd.DataFrame.from_dict(entity_types, orient="index", columns=["Count"])
    print(df)


if __name__ == "__main__":
    folder_path = 'dataset/exploded'
    file_paths = glob.glob(os.path.join(folder_path, '**', '*.dxf'), recursive=True)
    print(file_paths)
    for file in file_paths:
        print(f"counting {file}")
        count_entity_types('file')