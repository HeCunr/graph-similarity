import torch

# 定义实体类型索引
ENTITY_TYPES = [
    'LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT',
    'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT', 'EOS'
]

# 获取EOS的索引
EOS_IDX = ENTITY_TYPES.index('EOS')  # 应该是10，因为EOS在列表的最后

def _get_padding_mask(commands, seq_dim=0, extended=False):
    """
    获取填充掩码
    Args:
        commands: 命令序列张量
        seq_dim: 序列维度，默认为0
        extended: 是否扩展掩码，默认为False
    Returns:
        padding_mask: 填充掩码张量
    """
    with torch.no_grad():
        # 1表示非EOS，0表示EOS
        padding_mask = (commands == EOS_IDX).cumsum(dim=seq_dim) == 0
        padding_mask = padding_mask.float()

        if extended:
            # 扩展掩码以包含最后的EOS
            S = commands.size(seq_dim)
            torch.narrow(padding_mask, seq_dim, 3, S-3).add_(
                torch.narrow(padding_mask, seq_dim, 0, S-3).clone()
            ).clamp_(max=1)

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask  # (S, N, 1)

def _get_visibility_mask(commands, seq_dim=0):
    """
    获取可见性掩码
    Args:
        commands: 命令序列张量，形状为[S, ...]
        seq_dim: 序列维度，默认为0
    Returns:
        visibility_mask: 可见性掩码张量
    """
    S = commands.size(seq_dim)
    with torch.no_grad():
        # 检查是否序列中EOS的数量小于序列长度减1
        visibility_mask = (commands == EOS_IDX).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask

# 如果需要，可以添加其他类型的索引
LINE_IDX = ENTITY_TYPES.index('LINE')
CIRCLE_IDX = ENTITY_TYPES.index('CIRCLE')
ARC_IDX = ENTITY_TYPES.index('ARC')
LWPOLYLINE_IDX = ENTITY_TYPES.index('LWPOLYLINE')
TEXT_IDX = ENTITY_TYPES.index('TEXT')
MTEXT_IDX = ENTITY_TYPES.index('MTEXT')
HATCH_IDX = ENTITY_TYPES.index('HATCH')
DIMENSION_IDX = ENTITY_TYPES.index('DIMENSION')
LEADER_IDX = ENTITY_TYPES.index('LEADER')
INSERT_IDX = ENTITY_TYPES.index('INSERT')