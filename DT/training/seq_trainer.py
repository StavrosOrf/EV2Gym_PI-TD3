import numpy as np
import torch

from DT.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, action_mask = self.get_batch(
            self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
            action_mask=action_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1,
                                            act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1,
                                              act_dim)[attention_mask.reshape(-1) > 0]
        # print(f'state_target: {state_target.shape}')
        # print(f'action_target: {action_target.shape}')
        
        state_target = state_target.reshape(-1, states.shape[2])[
            attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            state_target, action_target, None
        )
        # print(f'Loss: {loss}')

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean(
        #         (action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
