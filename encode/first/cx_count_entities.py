# !/user/bin/env python3
# -*- coding: utf-8 -*-
import glob
import math
from collections import defaultdict
import os
import ezdxf
import pandas as pd
from ezdxf.entities import BoundaryPathType, EdgeType

import change1

def countAndRead_insert_types(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    entity_types = {}
    insert_types = []
    insert_count = {}
    i = 0
    for entity in modelspace:
        entity_type = entity.dxftype()
        if entity_type == 'INSERT':
            INSERTname = entity.dxf.name
            INSERTposition = entity.dxf.insert
            INSERTinfo = [INSERTname, INSERTposition, entity.dxf.xscale, entity.dxf.yscale, entity.dxf.rotation]
            insert_types.append(INSERTinfo)

        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

    for i in insert_types:
        print(i)
        insert_count[i[0]] = insert_count.get(i[0], 0) + 1
    print(insert_count)
    for item, count in insert_count.items():
        if count >= 3:
            print(f"{item}: {count}")


    df = pd.DataFrame.from_dict(entity_types, orient="index", columns=["Count"])
    print(df)
    print(entity_types)
    return insert_types, insert_count

if __name__ == '__main__':
    file_path = r'C:\srtp\encode\datasets\7.dxf'
    countAndRead_insert_types(file_path)
