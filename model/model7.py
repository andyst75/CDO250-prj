import torch
import torch.nn as nn
import torch.nn.functional as F


PREVIOUS = 360
PREDICT = 90
MODEL_NAME = "model7.pth"


class Model(nn.Module):
    def __init__(self, device, predicts=PREDICT, v_dim=PREVIOUS, e_dim=16, ff_dim=512):
        super(Model, self).__init__()

        self.v_dim = v_dim
        self.e_dim = e_dim
        self.predicts = predicts
        self.device = device
        
        self.embDay        = nn.Embedding(31, e_dim)
        self.embMonth      = nn.Embedding(12, e_dim)
        self.embWeekOfYear = nn.Embedding(53, e_dim)
        self.embWeekDay    = nn.Embedding(7, e_dim)
        self.embLayer      = nn.Linear(4 * e_dim * v_dim, ff_dim)
        self.embNorm       = nn.LayerNorm(ff_dim)
        self.emb1_dropout  = nn.Dropout(0.05)

        self.temp          = nn.Linear(v_dim, ff_dim)
        self.tempNorm      = nn.LayerNorm(ff_dim)
        self.use_fact      = nn.Linear(v_dim, ff_dim)
        self.useNorm       = nn.LayerNorm(ff_dim)
        self.consume       = nn.Linear(v_dim, ff_dim)
        self.consNorm      = nn.LayerNorm(ff_dim)
        
        self.emb2_dropout  = nn.Dropout(0.05)
        self.to_embs       = nn.Linear(ff_dim * 3, ff_dim)
        
        self.ff_1           = nn.Linear(ff_dim  + ff_dim, ff_dim * 2)
        self.ff_2           = nn.Linear(ff_dim * 2, ff_dim * 2)
        self.ff_3           = nn.Linear(ff_dim * 2, ff_dim * 2)
        self.ff_4           = nn.Linear(ff_dim * 2, ff_dim * 2)
        self.ff_5           = nn.Linear(ff_dim * 2, ff_dim * 2)
        self.ff_6           = nn.Linear(ff_dim * 2, ff_dim * 2)

        self.ff             = nn.Linear(ff_dim * 2, ff_dim * 4)
        
        self.fc             = nn.Linear(ff_dim * 4, predicts)

    def forward(self, day, month, weekofyear, weekday,
                temp, use_fact, gen_fact, consume, **kwargs):
        
        embDay = self.embDay(day.transpose(0, 1)).transpose(0, 1).flatten(start_dim=1)
        embMonth = self.embMonth(month.transpose(0, 1)).transpose(0, 1).flatten(start_dim=1)
        embWeekOfYear = self.embWeekOfYear(weekofyear.transpose(0, 1)).transpose(0, 1).flatten(start_dim=1)
        embWeekDay = self.embWeekDay(weekday.transpose(0, 1)).transpose(0, 1).flatten(start_dim=1)
        
        emb = torch.hstack([embDay, embMonth, embWeekOfYear, embWeekDay])
        emb = self.embLayer(emb)
        emb = self.embNorm(emb)
        emb = self.emb1_dropout(emb)
        
        mean_use_fact = use_fact.mean(axis=1).detach().view(-1, 1)
        
        if self.training:
            k = 0.022
            use_fact = use_fact * (1 + torch.randn(use_fact.size()).to(self.device) * k)
            mean_use_fact = mean_use_fact * (1 + torch.randn(mean_use_fact.size()).to(self.device) * k)
            temp = temp * (1 + torch.randn(temp.size()).to(self.device) * k)
        
        values = torch.hstack([
            self.tempNorm(self.temp(temp)),
            self.useNorm(self.use_fact(use_fact)),
            self.consNorm(self.consume(consume))
        ])
        values = self.to_embs(F.leaky_relu(values))
        values = self.emb2_dropout(values)
        
        out = torch.hstack([values, emb])

        out = self.ff_1(F.relu(out))
        out += self.ff_2(F.relu(out))
        out += self.ff_3(F.relu(out))
        out += self.ff_4(F.relu(out))
        out += self.ff_5(F.relu(out))
        out += self.ff_6(F.relu(out))

        out = self.ff(F.relu(out))
        
        out = mean_use_fact * (1 + out)
        
        out = self.fc(F.leaky_relu(out))

        return out