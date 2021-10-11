import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import json
import os
from tqdm import tqdm, trange

from video_summary.layers.summarizer import Summarizer
from video_summary.layers.discriminator import Discriminator
from video_summary.layers.actor_critic import Actor, Critic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        '''
        Class that builds, trains and evaluates the model
        '''

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

    def build(self):

        # Modules
        self.linear_compress = nn.Linear(
            self.config.input_size,
            self.config.hidden_size
        ).to(device=device)

        self.summarizer = Summarizer(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        ).to(device=device)

        self.discriminator = Discriminator(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers
        ).to(device=device)

        self.actor = Actor(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size
        ).to(device=device)

        self.critic = Critic(
            state_size=self.config.action_state_size,
            action_size=self.config.action_state_size
        ).to(device=device)

        self.model = nn.ModuleList([
            self.linear_compress,
            self.summarizer,
            self.discriminator,
            self.actor,
            self.critic
        ])

    def load_model(self, path):
        '''
        Load state_dict from a pretrained model
        '''

        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def AC(self, original_features, seq_len, action_fragments):
        '''
        Function that makes the actor's actions, in the training steps where the actor and critic components are not trained
        Args:
            original_features: [seq_len, 1, hidden_size]
            seq_len: scalar
            action_fragments: [action_state_size, 2]
        Return:
            weighted_features: [seq_len, 1, hidden_size]
            weighted_scores: [seq_len, 1]
        '''

        # [seq_len, 1]
        scores = self.summarizer.s_lstm(original_features)

        # [num_fragments, 1]
        fragment_scores = np.zeros(self.config.action_state_size)
        for fragment in range(self.config.action_state_size):
            fragment_scores[fragment] = scores[action_fragments[fragment, 0]:action_fragments[fragment, 1] + 1].mean()
        state = fragment_scores

        previous_actions = []    # save all the actions (the selected fragments of each episode)
        reduction_factor = (self.config.action_state_size - self.config.termination_point)/self.config.action_state_size
        action_scores = (torch.ones(seq_len)*reduction_factor).to(device=device)
        action_fragment_scores = (torch.ones(self.config.action_state_size)).to(device=device)

        counter = 0
        for ACstep in range(self.config.termination_point):

            state = torch.FloatTensor(state).to(device=device)

            # select an action
            dist = self.actor(state)
            action = dist.sample()  # return a scalar between 0 - action_state_size

            if action not in previous_actions:
                previous_actions.append(action)
                action_factor = (self.config.termination_point - counter)/(self.config.action_state_size - counter) + 1

                action_scores[action_fragments[action, 0]:action_fragments[action, 1] + 1] = action_factor
                action_fragment_scores[action] = 0

                counter += 1
            
            next_state = state*action_fragment_scores
            next_state = next_state.cpu().detach().numpy()
            state = next_state

        weighted_scores = action_scores.unsqueeze(1)*scores
        weighted_features = weighted_scores.view(-1, 1, 1)*original_features

        return weighted_features, weighted_scores

    def evaluate(self, epoch_i):

        self.model.eval()

        out_dict = {}

        for image_features, video_name, action_fragments in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            # [seq_len, batch_size=1, input_size]
            image_features = image_features.view(-1, self.config.input_size)
            image_features_ = Variable(image_features).to(device=device)

            # [seq_len, 1, hidden_size]
            original_features = self.linear_compress(image_features_.detach()).unsqueeze(1)
            seq_len = original_features.shape[0]
            
            with torch.no_grad():

                _, scores = self.AC(original_features, seq_len, action_fragments)

                scores = scores.squeeze(1)
                scores = scores.cpu().numpy().tolist()

                out_dict[video_name] = scores

            score_save_path = self.config.score_dir.joinpath(f'{self.config.video_type}_{epoch_i}.json')
            if not os.path.isdir(self.config.score_dir):
                os.makedirs(self.config.score_dir)

            with open(score_save_path, 'w') as f:
                if self.config.verbose:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                json.dump(out_dict, f)
            score_save_path.chmod(0o777)