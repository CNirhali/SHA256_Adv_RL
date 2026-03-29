import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class PPOActorCritic(nn.Module):
    def __init__(self, max_steps, action_dims):
        super(PPOActorCritic, self).__init__()
        
        # The input is the sequence of acts up to max_steps
        # flattened: max_steps * 4 features
        input_dim = max_steps * 4
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Critic network (Value estimator)
        self.critic = nn.Linear(128, 1)
        
        # Actor networks (One per action dimension)
        self.actor_ops = nn.Linear(128, action_dims[0])
        self.actor_tgt = nn.Linear(128, action_dims[1])
        self.actor_src1 = nn.Linear(128, action_dims[2])
        self.actor_src2 = nn.Linear(128, action_dims[3])

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten
        shared_features = self.shared(x)
        
        value = self.critic(shared_features)
        
        logits_ops = self.actor_ops(shared_features)
        logits_tgt = self.actor_tgt(shared_features)
        logits_src1 = self.actor_src1(shared_features)
        logits_src2 = self.actor_src2(shared_features)
        
        return value, (logits_ops, logits_tgt, logits_src1, logits_src2)
        
    def get_action(self, x):
        value, logits_tuple = self.forward(x)
        
        dists = [Categorical(logits=logits) for logits in logits_tuple]
        actions = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]
        
        action_tensor = torch.stack(actions, dim=-1)
        log_prob_tensor = torch.stack(log_probs, dim=-1).sum(dim=-1)
        
        return action_tensor, log_prob_tensor, value
