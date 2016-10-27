ADDITIVE REINFORCEMENT LEARNING:

- IDEA:
  - reuse pretrained skills
  - compose superagent from ready-to-use units
  - learn superagent's driver to manipulate with units
  - method: warp unit's observations
  - problem: detect and isolate external and internal observations

- PROFIT:
  - rapid learning
  - reusable components
  - additive learning
  - rectifiable learning procces (introducing compensation to achive goals by new ways open by driver)
  - transfer the skills as weight from one agent to other (from arm to leg, left to righ etc) and tune via driver

- CHALLENGE:
  - utilize the human-mentors intuition to improve the RL preformance
  - separate ext/eig observations to generate explicit tasks between units
  - learn by samples -- looking for profy and try to repeat the solution by self (mirror neurons)
  - teaching -- teacher looks for student's performance and prepares meta parameters and new cases to improve it

- QUESTIONS:
  - which alg use for the superagent's driver
  - how to speed up the unit learning
  - how to obtain the feedback from unit
  - how to train unit simultaniously with its driver
  - how to avoid local minimum (bad habits) e.g. far goals instead of close ones when the real goal is near a floor
  - add motivation to driver -- avoid goals' trimmer
  - other motivations?
  - problem -- need for quick training to make more experiments. How to solve?
  - must driver have more large hidden layers then the unit's net driven by it?
  - how much should be the noise range relatevely to the action signal
  - how to normalize the actions and obs to make them laying in the [0..1] interval
  - how to hint the agent what to do (m/b via motivation?)
  - how to tune the loss function automatically

