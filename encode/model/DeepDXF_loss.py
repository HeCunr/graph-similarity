import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature=0.5, loss_type='simclr'):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.loss_type = loss_type
        self.register_buffer('mask', self.create_mask(batch_size))

    def create_mask(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=torch.float32)
        mask.fill_diagonal_(0)
        return mask

    def forward(self, z1, z2):
        if self.loss_type == 'simclr':
            return self.simclr_loss(z1, z2)
        elif self.loss_type == 'infonce':
            return self.infonce_loss(z1, z2)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def simclr_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.mask.to(z1.device) * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

    def infonce_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = torch.sum(torch.exp(similarity_matrix / self.temperature) * self.mask.to(z1.device), dim=1)

        loss = -torch.log(nominator / denominator)
        loss = loss.mean()
        return loss