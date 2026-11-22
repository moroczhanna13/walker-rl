import gymnasium as gym


class PushWrapper(gym.Wrapper):
    """Időnként vízszintesen meglöki a robot törzsét.

    force:    a lökés ereje newtonban (0 = nincs lökés)
    interval: átlagosan hány lépésenként jön lökés
    duration: hány lépésig tart egy lökés
    side:     0 = véletlen irány, +1 = mindig hátulról (előre lök), -1 = mindig elölről
    """

    def __init__(self, env, force=100.0, interval=200, duration=5, side=0):
        super().__init__(env)
        self.force = force
        self.interval = interval
        self.duration = duration
        self.side = side
        self.torso_id = env.unwrapped.model.body("torso").id
        self.steps_left = 0          # ennyi lépés van még hátra az aktuális lökésből
        self.direction = 1.0
        self.push_count = 0

    def reset(self, **kwargs):
        self.steps_left = 0
        self.push_count = 0
        self.unwrapped.data.xfrc_applied[self.torso_id, :] = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Minden lépésben 1/interval eséllyel új lökés indul
        if self.force > 0 and self.steps_left == 0 and self.np_random.random() < 1.0 / self.interval:
            self.steps_left = self.duration
            self.direction = self.side if self.side != 0 else self.np_random.choice([-1.0, 1.0])
            self.push_count += 1

        # xfrc_applied: külső erő egy testre; a 0. elem az x irányú (előre-hátra) erő
        pushing = self.steps_left > 0
        self.unwrapped.data.xfrc_applied[self.torso_id, 0] = self.direction * self.force if pushing else 0.0
        if pushing:
            self.steps_left -= 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["pushing"] = pushing
        info["push_count"] = self.push_count
        return obs, reward, terminated, truncated, info

class PhysicsWrapper(gym.Wrapper):
    """Megváltoztatja a robot fizikáját. Minden érték szorzó: 1.0 = eredeti.

    mass:      a törzs tömege (2.0 = kétszer olyan nehéz törzs)
    friction:  a talaj és a lábak súrlódása (0.3 = csúszós, jeges)
    motor:     a motorok ereje (0.7 = 30%-kal gyengébb motorok)
    randomize: ha nagyobb 0-nál, minden epizód elején mindhárom szorzót
               véletlenül húzza az [1 - randomize, 1 + randomize] tartományból
    """

    def __init__(self, env, mass=1.0, friction=1.0, motor=1.0, randomize=0.0):
        super().__init__(env)
        self.scales = dict(mass=mass, friction=friction, motor=motor)
        self.randomize = randomize
        model = env.unwrapped.model
        self.torso_id = model.body("torso").id
        # Az eredeti értékeket elmentjük, mindig ezekből számolunk
        self.nominal_mass = model.body_mass.copy()
        self.nominal_inertia = model.body_inertia.copy()
        self.nominal_friction = model.geom_friction.copy()
        self.nominal_gear = model.actuator_gear.copy()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.randomize > 0:
            low, high = 1 - self.randomize, 1 + self.randomize
            self.scales = {name: self.np_random.uniform(low, high) for name in self.scales}

        model = self.unwrapped.model
        model.body_mass[self.torso_id] = self.nominal_mass[self.torso_id] * self.scales["mass"]
        model.body_inertia[self.torso_id] = self.nominal_inertia[self.torso_id] * self.scales["mass"]
        model.geom_friction[:, 0] = self.nominal_friction[:, 0] * self.scales["friction"]
        model.actuator_gear[:, 0] = self.nominal_gear[:, 0] * self.scales["motor"]
        info["physics"] = dict(self.scales)
        return obs, info