import GeneralClassifier
from GeneralClassifier import *

class SimpleClassifier(GeneralClassifier):

    def __init__(self, device, num_classes=1):
        super(SimpleClassifier, self).__init__()

        self.device = device

        # Number of input features is 12.
        # self.layer_1 = nn.Linear(input_dim, 1)
        # self.act1 = nn.ReLU()
        self.layer_out = nn.Linear(2, 1)
        self.out = nn.Sigmoid()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, y, genotypes_batch, beta, label=None, testing=False):

        y = torch.tensor(y).to(self.device).float()
        beta = torch.Tensor(beta).to(self.device)[:, None].float()
        genotypes_batch = torch.tensor(genotypes_batch).to(self.device).float()

        comparison_vector = torch.matmul(genotypes_batch, beta)

        # beta_summaries = self.act1(self.layer_1(beta))

        # x = torch.cat([y[:, None], torch.repeat_interleave(beta_summaries[None, :], y.shape[0], dim=0)], dim=1)
        x = torch.cat([y[:, None], comparison_vector], dim=-1)

        x = self.layer_out(x)

        logits = self.out(x)

        if label is not None:
            probs = torch.exp(logits) + 1e-6
            if testing:
                labels = probs.clone().squeeze()
                labels[labels >= 0.5] = 1
                labels[labels < 0.5] = 0
                print(torch.sum(torch.abs(labels - label)), "Wrong Preds out of", len(label))

                return [], []

            loss = self.criterion(probs, label[:, None])
            return probs, loss

        return torch.exp(logits[:, -1]) + 1e-6


