BUFFER_SIZE = 100 * 1000
BATCH_SIZE = 64
GAMMA = 0.99  # FUTURE REWARD DECAY
TAU = 0.001  # TARGET NETWORK UPDATE STEP
LRA = 0.0001  # ACTOR LEARNING_RATE
LRC = 0.001  # CRITIC LEARNING_RATE
L2C = 0.01  # L2 REGULARISATION CRITIC
L2A = 0.  # L2 REGULARISATION ACTOR
