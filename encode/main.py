import torch
from model.CGMN_DeepDXF_model import CGMN_DeepDXF
from dxf_process import dxf_to_json, dxf_to_matrix

def calculate_similarity(dxf_file1, dxf_file2, model_path):
    # 加载预训练的CGMN_DeepDXF模型
    model = CGMN_DeepDXF(args)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 处理DXF文件
    json1 = dxf_to_json(dxf_file1)
    json2 = dxf_to_json(dxf_file2)

    geometric_matrix1, adjacency_matrix1 = dxf_to_matrix(dxf_file1)
    geometric_matrix2, adjacency_matrix2 = dxf_to_matrix(dxf_file2)

    # 准备模型输入
    batch_x_p1 = torch.tensor(geometric_matrix1).unsqueeze(0)
    batch_adj_p1 = torch.tensor(adjacency_matrix1).unsqueeze(0)
    entity_type1, entity_params1 = process_json(json1)

    batch_x_p2 = torch.tensor(geometric_matrix2).unsqueeze(0)
    batch_adj_p2 = torch.tensor(adjacency_matrix2).unsqueeze(0)
    entity_type2, entity_params2 = process_json(json2)

    # 计算相似度
    with torch.no_grad():
        z1, proj_z1_cgmn, proj_z1_deepdxf = model(batch_x_p1, batch_adj_p1, entity_type1, entity_params1)
        z2, proj_z2_cgmn, proj_z2_deepdxf = model(batch_x_p2, batch_adj_p2, entity_type2, entity_params2)

        cgmn_similarity = torch.cosine_similarity(proj_z1_cgmn, proj_z2_cgmn, dim=1)
        deepdxf_similarity = torch.cosine_similarity(proj_z1_deepdxf, proj_z2_deepdxf, dim=1)

        similarity = args.cgmn_weight * cgmn_similarity + args.deepdxf_weight * deepdxf_similarity

    return similarity.item()

def process_json(json_data):
    # 将JSON数据处理为模型输入格式
    # ...
    return entity_type, entity_params

if __name__ == "__main__":
    dxf_file1 = "example1.dxf"
    dxf_file2 = "example2.dxf"
    model_path = "CGMN_DeepDXF.pth"

    similarity = calculate_similarity(dxf_file1, dxf_file2, model_path)
    print(f"Similarity between {dxf_file1} and {dxf_file2}: {similarity:.4f}")