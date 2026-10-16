"""
CALM: Continual Associative Learning Model.

An agent that is never "trained then deployed": it perceives, predicts, acts,
observes the result and updates itself on every single time step.

Units / modules (see report for the full design):
  1. Sensory feature units     x = [colour one-hot (4), shape one-hot (4), bias]
  2. Outcome units             valence unit (sigmoid) + 4 sound units (softmax),
                               driven by associative weights W_v, W_s from x
  3. Associability units       u_f, one per feature: running |prediction error|.
                               They set each feature's learning rate
                               (Pearce-Hall) and the agent's curiosity.
  3b. Change detector          a global "unexpected uncertainty" signal (fast vs
                               slow average of surprise, like a noradrenergic
                               signal). When the world changes it raises all
                               associabilities, weakens old visit counts and
                               makes replay favour recent memories.
  4. Co-occurrence memory      A[type, type], learned with a competitive Hebbian
                               (instar) rule from what is seen together, with no reward.
  5. Episodic buffer           recent experiences replayed between steps
                               (hippocampus -> cortex consolidation).
  6. Action selection          value + curiosity bonus - distance, softmax policy;
                               when nothing is worth approaching, it explores
                               less-visited places (spatial novelty).

The flags in `CALMConfig` turn each mechanism off for the ablation runs.
"""
from dataclasses import dataclass
import numpy as np

from environment import N_COLORS, N_SHAPES, N_TYPES, type_features

N_FEAT = N_COLORS + N_SHAPES + 1


def features(t):
    c, s = type_features(t)
    x = np.zeros(N_FEAT)
    x[c] = 1.0
    x[N_COLORS + s] = 1.0
    x[-1] = 1.0
    return x


FEATS = np.stack([features(t) for t in range(N_TYPES)])


@dataclass
class CALMConfig:
    alpha0: float = 0.10          # base learning rate
    associability: bool = True    # surprise-modulated, per-feature learning rate
    change_detection: bool = True # global unexpected-uncertainty signal
    curiosity: float = 0.6        # weight of the uncertainty bonus (0 = off)
    replay: int = 4               # replay updates per real interaction (0 = off)
    replay_capacity: int = 200
    replay_recency: float = 0.02  # exponential recency weighting for replay sampling
    hebb_rate: float = 0.03       # co-occurrence (instar) learning rate
    view_radius: int = 3
    temperature: float = 0.25
    learning: bool = True         # False = frozen weights
    passive: bool = False         # True = picks targets at random (no active choice)


class CALMAgent:
    def __init__(self, cfg: CALMConfig, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.W_v = np.zeros(N_FEAT)                  # valence associations
        self.W_s = np.zeros((N_SHAPES, N_FEAT))      # sound associations
        self.u = np.ones(N_FEAT)                     # associability: novel = uncertain
        self.n_type = np.zeros(N_TYPES)              # interaction counts per type
        self.A = np.full((N_TYPES, N_TYPES), 1.0 / N_TYPES)   # co-occurrence memory
        self.buffer = []
        self.s_fast = self.s_slow = 0.5              # surprise averages for change detection
        self.change_signal = 0.0
        self.visits = None
        self.target = None

    # ----- predictions ----------------------------------------------------
    def p_edible(self, t):
        return 1.0 / (1.0 + np.exp(-FEATS[t] @ self.W_v))

    def sound_probs(self, t):
        z = self.W_s @ FEATS[t]
        e = np.exp(z - z.max())
        return e / e.sum()

    def uncertainty(self, t):
        active = FEATS[t] > 0
        return self.u[active].mean() + 1.0 / np.sqrt(1.0 + self.n_type[t])

    def predicted_partner(self, t):
        row = self.A[t].copy()
        row[t] = -np.inf
        return int(np.argmax(row))

    # ----- learning ------------------------------------------------------
    def _delta_update(self, t, edible, sound, scale=1.0, track=True):
        x = FEATS[t]
        if self.cfg.associability:
            lr = self.cfg.alpha0 * (0.25 + self.u) * x * scale   # per-feature rate
        else:
            lr = self.cfg.alpha0 * x * scale
        d_v = edible - self.p_edible(t)                          # valence prediction error
        target = np.zeros(N_SHAPES)
        target[sound] = 1.0
        d_s = target - self.sound_probs(t)                       # sound prediction error
        self.W_v += lr * d_v
        self.W_s += np.outer(d_s, lr)
        if track:   # associability follows recent surprise (Pearce-Hall)
            surprise = 0.5 * abs(d_v) + 0.5 * np.abs(d_s).sum() / 2
            active = x > 0
            self.u[active] = 0.8 * self.u[active] + 0.2 * surprise
        return abs(d_v)

    def learn_from_outcome(self, out):
        if not self.cfg.learning:
            return
        t = out["type"]
        self.n_type[t] += 1
        surprise = self._delta_update(t, out["edible"], out["sound"])

        # change detection: surprise well above its long-run level means the world changed
        self.s_fast = 0.9 * self.s_fast + 0.1 * surprise
        self.s_slow = 0.99 * self.s_slow + 0.01 * surprise
        excess = max(0.0, self.s_fast - self.s_slow - 0.05) if self.cfg.change_detection else 0.0
        self.change_signal = excess
        if excess > 0:
            self.u = np.minimum(self.u + 2.0 * excess, 2.0)    # everything becomes uncertain again
            self.n_type *= 1.0 - excess                         # old experience counts for less

        self.buffer.append((t, out["edible"], out["sound"]))
        if len(self.buffer) > self.cfg.replay_capacity:
            self.buffer.pop(0)
        if self.cfg.replay and len(self.buffer) > 1:
            age = np.arange(len(self.buffer))[::-1]
            w = np.exp(-self.cfg.replay_recency * (1 + 50 * excess) * age)
            idx = self.rng.choice(len(self.buffer), size=self.cfg.replay, p=w / w.sum())
            for i in idx:
                self._delta_update(*self.buffer[i], scale=0.5, track=False)

    def learn_cooccurrence(self, visible):
        """Instar Hebbian rule: row for type i moves toward what is seen near i."""
        if not self.cfg.learning or len(visible) < 2:
            return
        for a in visible:
            ta = a["type"]
            post = np.zeros(N_TYPES)
            for b in visible:
                if b is a:
                    continue
                d = np.abs(a["pos"] - b["pos"]).max()
                post[b["type"]] += np.exp(-2.0 * (d - 1))              # closer = stronger
            if post.sum() == 0:
                continue
            post /= post.sum()
            self.A[ta] += self.cfg.hebb_rate * (post - self.A[ta])

    # ----- acting --------------------------------------------------------
    def act(self, env):
        """One perceive-predict-act step. Returns (move, object_to_interact_with)."""
        if self.visits is None:
            self.visits = np.zeros((env.size, env.size))
        self.visits *= 0.995
        self.visits[tuple(env.agent_pos)] += 1

        vis = env.visible(self.cfg.view_radius)
        self.learn_cooccurrence(vis)

        target = None
        if vis:
            if self.cfg.passive:
                if not any(o is self.target for o in vis):
                    self.target = vis[self.rng.integers(len(vis))]
                target = self.target
            else:
                scores = []
                for o in vis:
                    value = 2 * self.p_edible(o["type"]) - 1
                    bonus = self.cfg.curiosity * self.uncertainty(o["type"])
                    dist = np.abs(o["pos"] - env.agent_pos).sum()
                    sticky = 0.2 if o is self.target else 0.0
                    scores.append(value + bonus - 0.05 * dist + sticky)
                scores = np.array(scores)
                if scores.max() > -0.15:          # something is worth approaching
                    p = np.exp((scores - scores.max()) / self.cfg.temperature)
                    target = vis[self.rng.choice(len(vis), p=p / p.sum())]
        self.target = target

        if target is not None:
            move = np.sign(target["pos"] - env.agent_pos)
            if (env.agent_pos + move == target["pos"]).all() or (move == 0).all():
                return move, target
            return move, None
        return self._explore_move(env), None

    def _explore_move(self, env):
        moves = [np.array(m) for m in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]]
        if self.cfg.curiosity > 0 and not self.cfg.passive:
            # spatial novelty: head toward the least-visited neighbouring cell
            costs = []
            for m in moves:
                p = np.clip(env.agent_pos + m * 2, 0, env.size - 1)
                costs.append(self.visits[tuple(p)] + 0.1 * self.rng.random())
            return moves[int(np.argmin(costs))]
        return moves[self.rng.integers(len(moves))]
