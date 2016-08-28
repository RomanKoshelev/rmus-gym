#!/usr/bin/env python
import Box2D
import gym
import numpy as np
# noinspection PyUnresolvedReferences
from Box2D.b2 import edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener
from gym import spaces
from gym.utils import seeding

# region Configuration
FPS = 25
SCALE = 30.0
VIEWPORT_W = 900
VIEWPORT_H = 600
GROUND_HEIGHT = VIEWPORT_H / SCALE / 6
GROUND_MASK = 0x001
SEGMENT_MASK = 0x002
TARGET_MASK = 0x004

# endregion


class Tentacle(gym.Env):
    # --------------------------------------------------------------------------------------------------------------
    # region Initiation

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self):
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.ground = None
        self.tentacle = None

        self.action_space = spaces.Discrete(2)
        max_ob = np.array([1., 1.])
        self.observation_space = spaces.Box(-max_ob, max_ob)

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Creation

    def _reset(self):
        self.drawlist = []
        W, H = self._world_size()

        target_pos = (W / 2 - 3, GROUND_HEIGHT + 7)
        self._create_ground()
        self._create_tentacle()
        self._create_target(*target_pos)
        self.drawlist = self.ground + self.tentacle + self.target
        return self._step(0)[0]

    def _create_ground(self):
        self.ground = []
        shape = [(-100, -100), (100, -100), (100, GROUND_HEIGHT), (-100, GROUND_HEIGHT)]
        f = 2.5
        t = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(vertices=shape),
                categoryBits=GROUND_MASK,
                friction=f
            ))
        t.color1 = (0., 0.4, 0.)
        t.color2 = (0., 0.5, 0.)
        self.ground.append(t)

    def _create_target(self, x, y):
        self.target = []
        r = .1
        t = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=circleShape(radius=r, pos=(x, y)),
                categoryBits=TARGET_MASK
            ))
        t.color1 = (0.9, 0., 0.)
        t.color2 = (0.4, 0., 0.)
        self.target.append(t)

    def _create_tentacle(self):
        self.tentacle = []
        self.joints = []

        segdef = [
            {'w': 4., 'h': 1., 'd': 5.},
            {'w': .5, 'h': 4., 'd': .5},
            {'w': .4, 'h': 3., 'd': .5},
            {'w': .3, 'h': 2., 'd': .5}
        ]

        s0 = self._segment(segdef[0])
        s1 = self._segment(segdef[1], parent=s0)
        s2 = self._segment(segdef[2], parent=s1)
        s3 = self._segment(segdef[3], parent=s2)

        j01 = self._joint(s0, s1)
        j12 = self._joint(s1, s2)
        j23 = self._joint(s2, s3)

        self.joints.extend([j01, j12, j23])
        self.tentacle.extend([s0, s1, s2, s3])

    def _segment(self, sdef, parent=None):
        W, H = self._world_size()
        w = sdef['w']
        h = sdef['h']
        x = W / 2 - w / 2
        y = GROUND_HEIGHT + 5 if parent is None else parent.ini['y'] + parent.ini['h']
        d = sdef['d']
        r = 0.1
        s = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (0, h), (w, h), (w, 0)]),
                density=d,
                categoryBits=SEGMENT_MASK,
                maskBits=GROUND_MASK,
                restitution=r
            ))
        s.color1, s.color2 = (0.4, 0.4, 0.4), (0.3, 0.3, 0.3)
        s.ini = {'x': x, 'y': y, 'w': w, 'h': h}
        return s

    def _joint(self, a, b):
        angle = 1.3
        torque = 2.5
        speed = .25
        return self.world.CreateJoint(revoluteJointDef(
            bodyA=a,
            bodyB=b,
            localAnchorA=(a.ini['w'] / 2, a.ini['h']),
            localAnchorB=(b.ini['w'] / 2, 0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=torque,
            motorSpeed=speed,
            lowerAngle=-angle,
            upperAngle=angle,
        ))

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Processing

    def _step(self, action):
        self.world.Step(timeStep=1.0 / FPS, velocityIterations=6, positionIterations=2)
        state = [0., 0.]
        reward = 0
        done = False
        return np.array(state), reward, done, {}

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Rendering

    def _world_size(self):
        return VIEWPORT_W / SCALE, VIEWPORT_H / SCALE

    def _render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        # viewer
        w, h = self._world_size()
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, w, 0, h)

        # draw sky
        self.viewer.draw_polygon([(0, 0), (w, 0), (w, h), (0, h)], color=(0., 0., 0.))

        # draw objects
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    # endregion

    pass  # endlass


# region main
def main():
    game = "Tentacle-v0"
    env = gym.make(game)
    env.monitor.start('/tmp/' + game, force=True)
    env.reset()

    steps = 0
    total_reward = 0
    a = 0

    while True:
        s, r, done, info = env.step(a)
        total_reward += r

        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1

        a = env.action_space.sample()

        if done:
            break


if __name__ == "__main__":
    main()
# endregion
