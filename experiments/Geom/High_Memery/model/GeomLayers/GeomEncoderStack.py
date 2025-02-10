#model/GeomLayers/GeomEncoderStack.py
import torch
import torch.nn as nn
from model.GeomLayers.GeomEncoderBlock import GeomEncoderBlock
from model.GeomLayers.GeomGGNNBlock import GeomGGNNBlock

class GeomEncoderStack(nn.Module):
    """
    多层堆叠的 (MPNN + Aggregator)，示例节点从 4096 -> 1024 -> 256 -> 64，
    并在最后加一个 GeomGGNNBlock 做进一步图卷积处理。
    """

    def __init__(self, args, d_model=256):
        super().__init__()
        # 这里 d_model = 256
        # 前三层使用 GeomEncoderBlock，每层 (MPNN + aggregator) 会让节点数缩减:
        #   4096 -> 1024 -> 256 -> 64
        # 而特征维度(即 d_model)不变，均为 256
        self.blocks = nn.ModuleList([
            GeomEncoderBlock(d_model, in_nodes=4096, out_nodes=1024),
            GeomEncoderBlock(d_model, in_nodes=1024, out_nodes=256),
            GeomEncoderBlock(d_model, in_nodes=256, out_nodes=64),
            # 额外叠加一个 GeomGGNNBlock
            # 注意这里传入 node_init_dims=d_model=256，意味着输入特征维度是 256
            # 该 Block 内部根据 args.filters / args.conv 做多层图卷积
            GeomGGNNBlock(node_init_dims=d_model, args=args)
        ])

    def forward(self, x, adj, mask):
        """
        x:    [B, N, d_model], 初始 N=4096
        adj:  [B, N, N]
        mask: [B, N]

        returns:
            x, adj, mask
            其中 x: [B, 64, d_model], adj: [B,64,64], mask: [B,64]，经过三次聚合后
            再经过最后的 GeomGGNNBlock，输出依旧是 [B,64,256] (维度不变，节点不变)
        """
        for block in self.blocks:
            if isinstance(block, GeomGGNNBlock):
                # GeomGGNNBlock 的 forward 返回 x (特征), 不修改 adj/mask
                x = block(x, adj, mask=mask)
            else:
                # GeomEncoderBlock 的 forward 返回 (pfeat, padj, pmask)
                x, adj, mask = block(x, adj, mask)
        return x, adj, mask