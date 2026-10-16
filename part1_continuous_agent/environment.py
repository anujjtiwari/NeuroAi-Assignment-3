"""
A small, non-stationary grid world used to test continual associative learning.

World facts the agent has to discover by interacting:
  * Objects are made of two features: colour (4 values) and shape (4 values),
    giving 16 object types (type id = colour * 4 + shape).
  * Valence: touching an object is rewarding (+1, "edible") or punishing
    (-1, "toxic"). P(edible) depends additively on colour and shape.
  * Sound: touching an object also produces one of 4 sounds, set by its shape.
  * Co-occurrence: objects spawn in partner pairs (type A next to type B),
    so objects that tend to appear together can be learned without reward.

Things that change over time (to test continual learning):
  * t = switch_step: the valence of two colours swaps (partial reversal) and
    the co-occurrence pairing is reshuffled. The sound rule stays the same,
    so we can check whether old, still-valid knowledge is kept.
  * t = novel_step: four colour/shape combinations that never appeared before
    start to spawn (we probe zero-shot generalisation to them before that).
  * Objects drift around, expire after a random lifetime and respawn.
"""
import numpy as np

N_COLORS, N_SHAPES = 4, 4
N_TYPES = N_COLORS * N_SHAPES
HELD_OUT = [c * N_SHAPES + c for c in range(N_COLORS)]          # (c, c) combos
SEEN = [t for t in range(N_TYPES) if t not in HELD_OUT]


def type_features(t):
    return t // N_SHAPES, t % N_SHAPES


class AssociativeWorld:
    def __init__(self, size=12, n_objects=20, switch_step=6000, novel_step=9000,
                 drift_prob=0.1, lifetime=(40, 120), seed=0):
        self.rng = np.random.default_rng(seed)
        self.size, self.n_objects = size, n_objects
        self.switch_step, self.novel_step = switch_step, novel_step
        self.drift_prob, self.lifetime = drift_prob, lifetime

        # valence rule: logit = 1.5 * (colour_value + shape_value)
        self.color_value_1 = np.array([1.0, 1.0, -1.0, -1.0])
        self.color_value_2 = np.array([1.0, -1.0, 1.0, -1.0])   # colours 1 and 2 swap
        self.shape_value = np.array([0.5, -0.5, 0.5, -0.5])
        self.sound_of_shape = np.arange(N_SHAPES)                # stable across phases

        # co-occurrence pairings among the 12 seen types; a new pairing after the switch
        self.pairs_1 = self._random_pairing(SEEN)
        self.pairs_2 = self._random_pairing(SEEN, avoid=self.pairs_1)
        self.pairs_novel = self._random_pairing(HELD_OUT)

        self.t = 0
        self.objects = []   # dicts: type, pos (np.array), ttl
        self.agent_pos = np.array([size // 2, size // 2])
        while len(self.objects) < self.n_objects:
            self._spawn_pair()

    # ----- rules ---------------------------------------------------------
    def _random_pairing(self, types, avoid=None):
        for _ in range(1000):
            perm = self.rng.permutation(types)
            pairing = {}
            for a, b in zip(perm[::2], perm[1::2]):
                pairing[int(a)], pairing[int(b)] = int(b), int(a)
            if avoid is None or all(pairing[k] != avoid[k] for k in pairing):
                return pairing
        return pairing

    @property
    def phase(self):
        return 1 if self.t < self.switch_step else 2

    def color_value(self):
        return self.color_value_1 if self.phase == 1 else self.color_value_2

    def p_edible(self, t):
        c, s = type_features(t)
        return 1.0 / (1.0 + np.exp(-1.5 * (self.color_value()[c] + self.shape_value[s])))

    def true_valence_sign(self, t):
        return 1 if self.p_edible(t) > 0.5 else 0

    def true_sound(self, t):
        return int(self.sound_of_shape[type_features(t)[1]])

    def partner(self, t):
        if t in HELD_OUT:
            return self.pairs_novel[t]
        return (self.pairs_1 if self.phase == 1 else self.pairs_2)[t]

    def spawnable_types(self):
        return SEEN + HELD_OUT if self.t >= self.novel_step else SEEN

    # ----- dynamics ------------------------------------------------------
    def _free_cell_near(self, pos, radius):
        for _ in range(50):
            p = np.clip(pos + self.rng.integers(-radius, radius + 1, 2), 0, self.size - 1)
            if not any((o["pos"] == p).all() for o in self.objects) and not (p == self.agent_pos).all():
                return p
        return None

    def _spawn_pair(self):
        a = int(self.rng.choice(self.spawnable_types()))
        b = self.partner(a)
        pa = self._free_cell_near(self.rng.integers(0, self.size, 2), 0)
        if pa is None:
            return
        self.objects.append({"type": a, "pos": pa, "ttl": int(self.rng.integers(*self.lifetime))})
        pb = self._free_cell_near(pa, 1)
        if pb is not None:
            self.objects.append({"type": b, "pos": pb, "ttl": int(self.rng.integers(*self.lifetime))})

    def visible(self, radius):
        """Objects within a (2r+1)x(2r+1) window around the agent."""
        return [o for o in self.objects if np.abs(o["pos"] - self.agent_pos).max() <= radius]

    def step(self, move, interact_with=None):
        """move: (dx, dy) in {-1,0,1}^2. interact_with: an object dict at the agent's cell."""
        outcome = None
        self.agent_pos = np.clip(self.agent_pos + np.asarray(move), 0, self.size - 1)
        if interact_with is not None and any(o is interact_with for o in self.objects) \
                and (interact_with["pos"] == self.agent_pos).all():
            t = interact_with["type"]
            edible = self.rng.random() < self.p_edible(t)
            outcome = {"type": t, "edible": int(edible), "reward": 1.0 if edible else -1.0,
                       "sound": self.true_sound(t)}
            self.objects = [o for o in self.objects if o is not interact_with]

        for o in self.objects:
            o["ttl"] -= 1
            if self.rng.random() < self.drift_prob:
                o["pos"] = np.clip(o["pos"] + self.rng.integers(-1, 2, 2), 0, self.size - 1)
        self.objects = [o for o in self.objects if o["ttl"] > 0]
        while len(self.objects) < self.n_objects:
            self._spawn_pair()
        self.t += 1
        return outcome
