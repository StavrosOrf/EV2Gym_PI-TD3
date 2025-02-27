import numpy as np
import torch
import time

from DT.training.trainer import Trainer


class TrajectoryTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, action_mask = self.get_batch(
            self.batch_size)

        K = states.shape[1]
        
        self.verbose = False
        if self.verbose:
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
            
            if self.verbose:
                print('- ' * 40)
                print(f'Iteration: {i}')
                print(f'states_new: {states_new.shape}')
                print(f'actions_new: {actions_new.shape}')
                print(f'rtg_new: {rtg_new.shape}')
                print(f'attention_mask_new: {attention_mask_new.shape}')
                print(f'timesteps_new: {timesteps_new.shape}')

            _, action_preds, _ = self.model.forward(
                states_new,
                actions_new,
                None,
                rtg_new,
                timesteps_new,
                attention_mask=attention_mask_new,
                action_mask=None
            )

            # just_last_step                      
            # if i == K//2 - 1:                
            #     loss = -self.loss_fn(action=action_preds[:, -1, :],
            #                         state=states_new[:, -1, :]).float().mean()
            #     break
            
            # whole_last_iteration
            # if i == K//2 - 1:                
            #     loss = -self.loss_fn(action=action_preds.view(-1, action_preds.shape[-1]),
            #                         state=states_new.view(-1, states_new.shape[-1])
            #                         ).float().mean()                                
            #     break
                
            
            # Every_step
            if i == 0:
                loss = -self.loss_fn(action=action_preds[:, -1, :],
                                    state=states_new[:, -1, :]).float().mean()
            else:
                loss -= self.loss_fn(action=action_preds[:, -1, :],
                                    state=states_new[:, -1, :]).float().mean()
                if i == K//2 - 1:
                    break
                

            actions_new = torch.cat([actions_new[:, 1:, :],
                                     action_preds[:, -1, :].unsqueeze(1)], dim=1)

            # print(f'--actions_new: {actions_new.shape}')

            state_new = self.transition_fn(
                state=states_new[:, -1, :],
                new_state=states_new[:, -1, :],
                action=action_preds[:, -1, :],
            )

            # print(f'--state_new: {state_new.shape}')
            reward = self.loss_fn(action=action_preds[:, -1, :],
                                  state=state_new).float()
            
            if self.verbose:
                # print(f'--reward: {reward}')
                print(f'--reward: {reward.shape}')
                print(f'--rtg_new: {rtg_new.shape}')
                # print(f'--rtg_new: {rtg_new}')                                

            rtg_new = torch.cat([rtg_new[:, 1:, :].reshape(-1, K//2 - 1, 1),
                                 (rtg_new[:, -1, :] + reward.unsqueeze(1)).unsqueeze(1)
                                 ],
                                dim=1)

            states_new = torch.cat([states_new[:, 1:, :],
                                    state_new.unsqueeze(1)], dim=1)
            timesteps_new = torch.cat([timesteps_new[:, 1:],
                                       timesteps[:, K//2 + i].unsqueeze(1)], dim=1)
            attention_mask_new = torch.cat([attention_mask_new[:, 1:],
                                            attention_mask[:, K//2 + i].unsqueeze(1)], dim=1)

            # print(f'--timesteps_new: {timesteps_new}')
            # print(f'--rtg_new: {rtg_new}')
            # input('Press Enter to continue...')



        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()
