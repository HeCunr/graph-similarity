# !/user/bin/env python3
# -*- coding: utf-8 -*-
import ezdxf
import os

def get_layer_properties(file_path):
    doc = ezdxf.readfile(file_path)
    modelspace = doc.modelspace()

    layer_properties = {}

    for entity in modelspace:
        layer_name = entity.dxf.layer
        color = entity.dxf.color
        linetype = entity.dxf.linetype

        if layer_name not in layer_properties:
            layer_properties[layer_name] = {
                'colors': set(),
                'linetypes': set()
            }

        layer_properties[layer_name]['colors'].add(color)
        layer_properties[layer_name]['linetypes'].add(linetype)

    return layer_properties

def print_layer_properties(layer_properties):
    for layer, properties in layer_properties.items():
        colors = properties['colors']
        linetypes = properties['linetypes']
        print(f"Layer: {layer}")
        print(f"  Colors: {colors}")
        print(f"  Linetypes: {linetypes}")

if __name__ == '__main__':
    # 设置你的DXF文件路径
    file_path = r'C:\srtp\datasets\test\one - 副本.dxf'

    layer_properties = get_layer_properties(file_path)
    print_layer_properties(layer_properties)
