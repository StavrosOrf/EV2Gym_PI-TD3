import numpy as np
import torch
import time

from DT.training.trainer import Trainer


class TrajectoryTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, action_mask = self.get_batch(
            self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)

        K = states.shape[1]
        print('=' * 80)
        print(f'K: {K}')
        print(f'Init states: {states.shape}')
        print(f'Init actions: {actions.shape}')
        print(f'Init rtg: {rtg.shape}')
        print(f'Init attention_mask: {attention_mask.shape}')
        print(f'Init timesteps: {timesteps.shape}')

        states_new = states[:, :K//2, :]
        actions_new = actions[:, :K//2, :]
        rtg_new = rtg[:, :K//2, :]
        timesteps_new = timesteps[:, :K//2]
        attention_mask_new = attention_mask[:, :K//2]

        for i in range(K//2):
            print('- ' * 40)
            print(f'Iteration: {i}')
            print(f'states_new: {states_new.shape}')
            print(f'actions_new: {actions_new.shape}')
            print(f'rtg_new: {rtg_new.shape}')

            _, action_preds, _ = self.model.forward(
                states_new,
                actions_new,
                None,
                rtg_new,
                timesteps_new,
                attention_mask=attention_mask_new,
                action_mask=None
            )

            actions_new = torch.cat([actions_new[:, 1:, :],
                                     action_preds[:, -1, :].unsqueeze(1)], dim=1)

            print(f'actions_new: {actions_new.shape}')

            state_new = self.transition_fn(
                state=states_new[:, -1, :],
                new_state=states_new[:, -1, :],
                action=action_preds[:, -1, :],
            )

            print(f'state_new: {state_new.shape}')

            reward = self.loss_fn(action=action_preds[:, -1, :],
                                  state=state_new)
            print(f'reward: {reward}')

            states_new = torch.cat([states_new[:, 1:, :],
                                    state_new.unsqueeze(1)], dim=1)
            timesteps_new = torch.cat([timesteps_new[:, 1:],
                                       timesteps[:, K//2 + i].unsqueeze(1)], dim=1)
            attention_mask_new = torch.cat([attention_mask_new[:, 1:],
                                            attention_mask[:, K//2 + i].unsqueeze(1)], dim=1)

            print(f'timesteps_new: {timesteps_new}')
            input('Press Enter to continue...')

        # act_dim = action_preds.shape[2]
        # action_preds = action_preds.reshape(-1,
        #                                     act_dim)[attention_mask.reshape(-1) > 0]
        # action_target = action_target.reshape(-1,
        #                                         act_dim)[attention_mask.reshape(-1) > 0]
        # # print(f'state_target: {state_target.shape}')
        # # print(f'action_target: {action_target.shape}')

        # state_target = state_target.reshape(-1, states.shape[2])[
        #     attention_mask.reshape(-1) > 0]

        loss = 0

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        # with torch.no_grad():
        #     self.diagnostics['training/action_error'] = torch.mean(
        #         (action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()
