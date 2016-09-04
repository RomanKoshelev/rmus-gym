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

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,))

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
        sh = self._head(parent=s3)
        self.tentacle.extend([s0, s1, s2, s3, sh])

        j01 = self._rev_joint(s0, s1)
        j12 = self._rev_joint(s1, s2)
        j23 = self._rev_joint(s2, s3)
        self._weld_joint(s3, sh)
        self.joints.extend([j01, j12, j23])

    def _head(self, parent):
        W, H = self._world_size()
        r = .12
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

    def _rev_joint(self, a, b):
        angle = 1.3
        torque = 2.2
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
        assert self.action_space.contains(action), "Invalid action %r (%s)" % (action, type(action))
        assert len(action) == len(self.joints), "Action size: %d, joints: %d" % (len(action), len(self.joints))

        for i in range(len(self.joints)):
            j = self.joints[0]
            a = action[i]
            j.motorSpeed = float(1000 * np.sign(a))
            j.maxMotorTorque = float(1000 * np.clip(np.abs(a), 0, 1))

        self.world.Step(timeStep=1.0 / FPS, velocityIterations=6, positionIterations=2)
        state = self._make_state()
        reward = self._make_reward(state)
        done = False
        return np.array(state), reward, done, {}

    def _make_state(self):
        head = self.tentacle[len(self.tentacle) - 1]

        hx, hy = head.position[0], head.position[1]
        tx, ty = self.target[0].position[0], self.target[0].position[1]
        target_dist = np.sqrt((hx - tx) ** 2 + (hy - ty) ** 2)

        hvx, hvy = head.linearVelocity.x, head.linearVelocity.y
        head_vel = np.sqrt(hvx ** 2 + hvy ** 2)

        return target_dist, head_vel

    def _make_reward(self, state):
        d = state[0] + .0001  # distance to target
        v = state[1] + .0001  # head velocity
        return 1 / d ** 4 + v / 10 - 1

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
            def str_arr(arr, tmpl):
                return '[ ' + ', '.join([tmpl % x for x in arr]) + ' ]'

            log = ''
            SEP = '   :   '
            log += "step: %3d" % step + SEP
            log += "act: %s" % str_arr(a, "%+4.2f") + SEP
            log += "state: %s" % str_arr(s, "%4.2f") + SEP
            log += "reward: %4.1f" % r
            print(log)
        step += 1

        if done or step > 3000:
            break


if __name__ == "__main__":
    main()
# endregion
