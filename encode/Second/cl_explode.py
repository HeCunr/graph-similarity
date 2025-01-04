import ezdxf
import os

def explode_dxf(dxf_file, output_dxf_file):
    doc = ezdxf.readfile(dxf_file)
    msp = doc.modelspace()

    for e in msp:
        if e.dxftype() == 'INSERT':
            block = doc.blocks.get(e.dxf.name)
            if block is not None:
                for entity in block:
                    new_entity = entity.copy()
                    new_entity.transform(e.matrix44())
                    msp.add_entity(new_entity)
            e.destroy()

    for lwpolyline in msp.query('LWPOLYLINE'):
        vertices = lwpolyline.get_points('xy')
        is_closed = lwpolyline.is_closed
        for i in range(len(vertices) - 1):
            start_point = vertices[i]
            end_point = vertices[i + 1]
            msp.add_line(start=start_point, end=end_point)

        if is_closed:
            start_point = vertices[lwpolyline.dxf.count - 1]
            end_point = vertices[0]
            msp.add_line(start=start_point, end=end_point)

        lwpolyline.destroy()

    for polyline in msp.query('POLYLINE'):
        vertices = polyline.vertices
        is_closed = polyline.is_closed
        for i in range(len(vertices) - 1):
            start_point = vertices[i].dxf.location
            end_point = vertices[i + 1].dxf.location
            msp.add_line(start=start_point, end=end_point)

        if is_closed:
            start_point = vertices[len(vertices) - 1].dxf.location
            end_point = vertices[0].dxf.location
            msp.add_line(start=start_point, end=end_point)

        polyline.destroy()

    for entity in msp.query('TEXT'):
        text_content = entity.dxf.text
        text_location = entity.dxf.insert
        text_height = entity.dxf.height

        msp.add_mtext(text_content, dxfattribs={
            'insert': text_location,
            'char_height': text_height
        })

        entity.destroy()

    doc.saveas(output_dxf_file)



if __name__ == "__main__":
    dxf_file = r'C:\srtp\test'
    output_dxf_file = r'C:\srtp\test\explode'
    for file in os.listdir(dxf_file):
        input_file = os.path.join(dxf_file, file)
        output_file = os.path.join(output_dxf_file, file)

        explode_dxf(input_file, output_file)