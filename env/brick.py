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


# endregion


class Brick(gym.Env):
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
        self.segments = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,))  # impulse x
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(2,))  # dx, vx

        self._reset()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Creation

    def _destroy(self):
        if not self.ground:
            return

        def destroy_body(arr):
            for b in arr:
                self.world.DestroyBody(b)

        destroy_body(self.ground)
        destroy_body(self.target)
        destroy_body(self.segments)

        for j in self.joints:
            self.world.DestroyJoint(j)

    def _reset(self):
        self._destroy()
        self.drawlist = []

        W, H = self._world_size()
        rw = np.random.random()
        target_pos = (rw*W, GROUND_HEIGHT + 1.5)

        self._create_ground()
        self._create_segments()
        self._create_target(*target_pos)
        self.drawlist = self.ground + self.segments + self.target
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

    def _create_target(self, x, y):
        self.target = []
        r = .1
        t = self.world.CreateStaticBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=circleShape(radius=r, pos=(0, 0)),
                categoryBits=TARGET_MASK
            ))
        t.color1 = (0.9, 0., 0.)
        t.color2 = (0.8, 0., 0.)
        self.target.append(t)

    def _create_segments(self):
        self.segments = []
        self.joints = []

        seg_def = [
            {'w': 4., 'h': 1., 'd': 5.}
        ]

        s0 = self._segment(seg_def[0])
        self.segments.extend([s0])

    def _segment(self, sdef, parent=None):
        W, H = self._world_size()
        w = sdef['w']
        h = sdef['h']
        x = W / 2 - w / 2
        ini_y = 2
        y = GROUND_HEIGHT + ini_y if parent is None else parent.ini['y'] + parent.ini['h']
        d = sdef['d']
        s = self.world.CreateDynamicBody(
            position=(x, y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(0, 0), (0, h), (w, h), (w, 0)]),
                density=d,
                categoryBits=SEGMENT_MASK,
                maskBits=GROUND_MASK,
                friction=0.003
            ))
        s.color1, s.color2 = (0.4, 0.4, 0.4), (0.3, 0.3, 0.3)
        s.ini = {'x': x, 'y': y, 'w': w, 'h': h}
        return s

    # endregion

    # --------------------------------------------------------------------------------------------------------------
    # region Processing

    def _step(self, action):
        assert self.action_space.contains(action), "Invalid action %r (%s)" % (action, type(action))
        SIDE_ENGINE_POWER = 100
        base = self.segments[0]

        impulse_pos = (base.position[0], base.position[1])
        ox = action[0]
        base.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER, 0), impulse_pos, True)

        self.world.Step(timeStep=1.0 / FPS, velocityIterations=6, positionIterations=2)
        state = self._make_state()
        reward = self._make_reward(state)
        done = False
        return np.array(state), reward, done, {}

    def _make_state(self):
        dx = self.target[0].position[0] - self.segments[0].worldCenter[0]
        vx = self.segments[0].linearVelocity[0]
        s = [dx, vx]
        return s

    def _make_reward(self, state):
        d = state[0]**2 + 1.e-10
        r = 1/d
        return r

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
    s = np.sign(state[0])
    v = state[1]
    d = abs(state[0])
    d = min(d, 10.)/10
    a = s*d - v/10
    return np.array([a])


def main():
    game = "Brick-v0"
    env = gym.make(game)
    env.monitor.start('/tmp/' + game, force=True)
    env.reset()
    step = 0

    a = env.action_space.sample()
    while True:
        s, r, done, info = env.step(a)
        a = sample_action(env, s)

        if step % 10 == 0 or done:
            def str_arr(arr, tmpl):
                return '[ ' + ', '.join([tmpl % x for x in arr]) + ' ]'

            log = ''
            SEP = '   :   '
            log += "step: %3d" % step + SEP
            log += "act: %s" % str_arr(a, "%+4.2f") + SEP
            log += "state: %s" % str_arr(s, "%+4.2f") + SEP
            log += "reward: %4.1f" % r
            print(log)
        step += 1

        if done:
            break


if __name__ == "__main__":
    main()
# endregion
