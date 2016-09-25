#!/usr/bin/env python
from __future__ import print_function
import Box2D
import gym

import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, contactListener, revoluteJointDef,
                      weldJointDef)
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

# ------------------------------------
MAX_SPEED = 1
MAX_TORQUE = 1000

SEGMENTS = [
    {'w': 4., 'h': 1., 'd': 50.},
    {'w': .5, 'h': 4., 'd': .5},
    {'w': .4, 'h': 3., 'd': .5},
    {'w': .3, 'h': 2., 'd': .5}
]

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

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-100, high=100, shape=(6,))

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Creation

    def _reset(self):
        if not self.ground:
            self.drawlist = []
            self._create_ground()
            self._create_tentacle()
            self._create_target()
            self.drawlist = self.ground + self.tentacle + self.target

        W, _ = self._world_size()
        self.target[0].position[0] = W / 2 - 4 + 8 * np.random.random()
        self.target[0].position[1] = GROUND_HEIGHT + 5 + 0 * np.random.random()

        return np.array(self._make_state())

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

    def _create_target(self):
        self.target = []
        r = .2
        t = self.world.CreateStaticBody(
            position=(0, 0),
            fixtures=fixtureDef(
                shape=circleShape(radius=r, pos=(0, 0)),
                categoryBits=TARGET_MASK
            ))
        t.color1 = (0.9, 0., 0.)
        t.color2 = (0.8, 0., 0.)
        self.target.append(t)

    def _create_tentacle(self):
        self.tentacle = []
        self.joints = []

        s0 = self._segment(SEGMENTS[0])
        s1 = self._segment(SEGMENTS[1], parent=s0)
        sh = self._head(parent=s1)
        self.tentacle.extend([s0, s1, sh])

        j01 = self._rev_joint(s0, s1, 0.33 * np.pi)
        self._weld_joint(s1, sh)
        self.joints.extend([j01])

    def _head(self, parent):
        W, H = self._world_size()
        r = .15
        x = W / 2
        y = parent.ini['y'] + parent.ini['h']
        s = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=circleShape(radius=r, pos=(0, 0)),
                density=0.1,
                categoryBits=SEGMENT_MASK,
                maskBits=GROUND_MASK
            ))
        s.color1, s.color2 = (0.0, 0.9, 0.), (0.0, 0.5, 0.)
        s.ini = {'x': x, 'y': y, 'w': 2 * r, 'h': 2 * r}
        return s

    def _segment(self, sdef, parent=None):
        W, H = self._world_size()
        w = sdef['w']
        h = sdef['h']
        x = W / 2 - w / 2
        y = GROUND_HEIGHT if parent is None else parent.ini['y'] + parent.ini['h']
        d = sdef['d']
        s = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (0, h), (w, h), (w, 0)]),
                density=d,
                categoryBits=SEGMENT_MASK,
                maskBits=GROUND_MASK,
                friction=.3,
                restitution=.3
            ))
        s.color1, s.color2 = (0.4, 0.4, 0.4), (0.3, 0.3, 0.3)
        s.ini = {'x': x, 'y': y, 'w': w, 'h': h}
        return s

    def _rev_joint(self, a, b, angle):
        return self.world.CreateJoint(revoluteJointDef(
            bodyA=a,
            bodyB=b,
            localAnchorA=(a.ini['w'] / 2, a.ini['h']),
            localAnchorB=(b.ini['w'] / 2, 0),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=0,
            motorSpeed=0,
            lowerAngle=-angle,
            upperAngle=angle,
        ))

    def _weld_joint(self, a, b):
        return self.world.CreateJoint(weldJointDef(
            bodyA=a,
            bodyB=b,
            localAnchorA=(a.ini['w'] / 2 + b.ini['w'] / 2, a.ini['h']),
            localAnchorB=(b.ini['w'] / 2, 0),
            referenceAngle=0
        ))

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Processing

    def _step(self, action):
        for i in range(len(self.joints)):
            j = self.joints[i]
            a = action[i]
            j.motorSpeed = float(MAX_SPEED * np.sign(a))
            j.maxMotorTorque = float(MAX_TORQUE * np.clip(np.abs(a), 0, 1))

        self.world.Step(timeStep=1.0 / FPS, velocityIterations=6, positionIterations=2)
        state = self._make_state()
        reward = self._make_reward(state, action)
        done = False
        # print_log(action, reward, state)
        return np.array(state), reward, done, {}

    def _make_state(self):
        head = self.tentacle[len(self.tentacle) - 1]
        hx, hy = head.position[0], head.position[1]
        tx, ty = self.target[0].position[0], self.target[0].position[1]
        target_dist = np.sqrt((hx - tx) ** 2 + (hy - ty) ** 2)

        j0_angle = self.joints[0].angle
        j0_speed = self.joints[0].speed / MAX_SPEED

        return target_dist, \
               np.cos(j0_angle), \
               np.sin(j0_angle), \
               j0_speed, \
               hx - tx, \
               hy - ty

    def _make_reward(self, state, action):
        (d, v) = state[0], state[1]
        cost_dist = d
        cost_act = abs(action[0])
        cost_vel = v
        cost = 100. * cost_dist + \
               10. * cost_act + \
               0. * cost_vel
        return -cost

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


# ------------------------------------------------------------------------------------------------------------------
# region main

def sample_action(env, state):
    assert env.observation_space.contains(state), "Invalid state %r (%s)" % (state, type(state))
    return env.action_space.sample()


def main():
    game = "Tentacle-v0"
    env = gym.make(game)
    env.monitor.start('/tmp/' + game, force=True)
    env.reset()
    step = 0

    a = env.action_space.sample()
    while True:
        s, r, done, info = env.step(a)
        a = sample_action(env, s)

        if step % 10 == 0 or done:
            print_log(a, r, s)
        step += 1

        if done or step > 3000:
            break


def print_log(a, r, s):
    def str_arr(arr, tmpl):
        return '[ ' + ', '.join([tmpl % x for x in arr]) + ' ]'

    log = ''
    SEP = '   :   '
    log += "act: %s" % str_arr(a, "%+4.2f") + SEP
    log += "state: %s" % str_arr(s, "%4.2f") + SEP
    log += "reward: %4.1f" % r
    print(log)


if __name__ == "__main__":
    main()
# endregion
