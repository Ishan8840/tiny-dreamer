import numpy as np
import optax
from flax import nnx
from train import shortcut_loss

@nnx.jit
def update(net, optimizer, z, actions, targets, reward_z, reward_actions, reward_targets, key):
    def objective(dynamics):
        _, terminal = dynamics.outcome_outputs(z[:, :-1], actions)
        reward, _ = dynamics.outcome_outputs(reward_z[:, :-1], reward_actions)
        terminal_loss = optax.sigmoid_binary_cross_entropy(terminal, targets[:, 1]).mean()
        reward_loss = optax.softmax_cross_entropy_with_integer_labels(reward, reward_targets[:, 0].astype('int32')).mean()
        outcome = terminal_loss + reward_loss
        return shortcut_loss(dynamics, z, actions, key) + outcome, outcome
    (_, outcome), grads = nnx.value_and_grad(objective, has_aux=True)(net)
    optimizer.update(net, grads)
    return outcome

def endpoint_pools(episodes):
    pools = []
    for positive in (True, False):
        pairs = []
        for index, ep in enumerate(episodes):
            ends = np.flatnonzero((ep['rewards'][4:] > 0) == positive) + 4
            pairs.append(np.column_stack((np.full(len(ends),index), ends)))
        pools.append(np.concatenate(pairs))
    return pools

def reward_batch(sources, rng):
    windows = []
    for kind, count in enumerate((16, 48)):
        available = [(episodes, pools[kind]) for episodes, pools in sources if len(pools[kind])]
        for episodes, pool in available:
            for i, end in pool[rng.integers(len(pool),size=count//len(available))]:
                ep = episodes[i]
                windows.append((ep['observations'][end-4:end+2],ep['actions'][end-4:end+1],
                                (ep['rewards'][end],ep['terminals'][end])))
    z,a,t = map(np.stack,zip(*windows))
    assert len(t)==64 and np.count_nonzero(t[:,0]>0)==16
    return z,a,t
