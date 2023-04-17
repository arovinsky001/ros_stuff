#!/usr/bin/python

import numpy as np
import torch
from torch.nn import functional as F

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import dcn, as_tensor


def train_from_buffer(agent, replay_buffer, validation_buffer=None, pretrain_samples=500,
                     save_agent=False, train_epochs=100, batch_size=500, meta=False):
    n_samples = min(pretrain_samples, replay_buffer.size)
    states = replay_buffer.states[:n_samples]
    actions = replay_buffer.actions[:n_samples]
    next_states = replay_buffer.next_states[:n_samples]

    ###
    states = np.tile(states, (2, 1))
    actions = np.tile(actions, (2, 1))
    next_states = np.tile(next_states, (2, 1))

    tmp = states[n_samples:, :2].copy()
    states[n_samples:, :2] = states[n_samples:, 3:5].copy()
    states[n_samples:, 3:5] = tmp

    tmp = actions[n_samples:, :2].copy()
    states[n_samples:, :2] = states[n_samples:, 2:].copy()
    states[n_samples:, 2:] = tmp

    tmp = next_states[n_samples:, :2].copy()
    next_states[n_samples:, :2] = next_states[n_samples:, 3:5].copy()
    next_states[n_samples:, 3:5] = tmp
    ###

    if validation_buffer is None:
        validation_buffer = replay_buffer

    training_losses, test_losses = train(
            agent, states, actions, next_states, validation_buffer,
            set_scalers=True, epochs=train_epochs, batch_size=batch_size,
            meta=meta,
    )

    if save_agent:
        agent.dump()

    training_losses = np.array(training_losses).squeeze()
    test_losses = np.array(test_losses).squeeze()

    print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
    print("MIN TEST LOSS:", test_losses.min())

    val_state, val_action, val_next_state = validation_buffer.sample(validation_buffer.buffer_capacity)
    val_state_delta = agent.dtu.compute_relative_delta_xysc(val_state, val_next_state)
    val_state, val_action, val_next_state = as_tensor(val_state, val_action, val_next_state)

    with torch.no_grad():
        agent.models[-1].eval()
        pred_state_delta = agent.models[-1](val_state, val_action, sample=False, delta=True)

    error = abs(pred_state_delta - val_state_delta)
    print("\nERROR MEAN:", error.mean(axis=0))
    print("ERROR STD:", error.std(axis=0))
    print("ERROR MAX:", error.max(axis=0)[0])
    print("ERROR MIN:", error.min(axis=0)[0])

    diffs = abs(val_state_delta)
    print("\nACTUAL MEAN:", diffs.mean(axis=0))
    print("ACTUAL STD:", diffs.std(axis=0))

    # fig, axes = plt.subplots(1, 4)
    # axes[0].plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
    # axes[1].plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")

    # axes[0].set_yscale('log')
    # axes[1].set_yscale('log')

    # axes[0].set_title('Training Loss')
    # axes[1].set_title('Test Loss')

    # for ax in axes:
    #     ax.grid()

    # axes[0].set_ylabel('Loss')
    # axes[1].set_xlabel('Epoch')
    # fig.set_size_inches(15, 5)

    # plt.show()

def train(agent, train_state, train_action, train_next_state, validation_buffer, epochs=5, batch_size=256, set_scalers=False, meta=False):
    train_state, train_action, train_next_state = as_tensor(train_state, train_action, train_next_state)

    val_state, val_action, val_next_state = validation_buffer.sample(validation_buffer.size)
    val_state_delta = agent.dtu.compute_relative_delta_xysc(val_state, val_next_state)
    val_state, val_action, val_next_state = as_tensor(val_state, val_action, val_next_state)

    for _, model in enumerate(agent.models):
        if set_scalers:
            model.set_scalers(train_state, train_action, train_next_state)

        train_losses, test_losses = [], []
        n_batches = int(np.ceil(len(train_state) / batch_size))

        # evaluate
        with torch.no_grad():
            model.eval()
            pred_state_delta = model(val_state, val_action, sample=False, delta=True)
            test_loss_mean = F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')

        test_losses.append(dcn(test_loss_mean))
        tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")

        print("\n\nTRAINING MODEL\n")
        for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
            # train
            if meta:
                train_loss_mean = model.update_meta(train_state, train_action, train_next_state)
            else:
                shuffle_idx = np.random.permutation(len(train_state))
                train_state, train_action, train_next_state = train_state[shuffle_idx], train_action[shuffle_idx], train_next_state[shuffle_idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * batch_size, (j + 1) * batch_size
                    batch_state, batch_action, batch_next_state = train_state[start:end], train_action[start:end], train_next_state[start:end]
                    train_loss_mean = model.update(batch_state, batch_action, batch_next_state)

            # evaluate
            with torch.no_grad():
                model.eval()
                pred_state_delta = model(val_state, val_action, sample=False, delta=True)
                test_loss_mean = F.mse_loss(pred_state_delta, val_state_delta, reduction='mean')

            train_losses.append(train_loss_mean)
            test_losses.append(dcn(test_loss_mean))
            tqdm.write(f"{i+1}: train loss: {train_loss_mean:.5f} | test loss: {test_loss_mean:.5f}")

    if meta:
        for g in model.optimizer.param_groups:
            # g['lr'] = model.update_lr
            g['lr'] = torch.clamp(model.update_lr, 1e-4, 1)
            print("\nUPDATE LEARNING RATE:", model.update_lr)

    model.eval()
    agent.trained = True

    return train_losses, test_losses
