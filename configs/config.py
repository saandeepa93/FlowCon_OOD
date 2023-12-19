from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.DATA_ROOT = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.CKP = "~/Desktop/projects/dataset/AffectNet/"
_C.PATHS.VIS_PATH = ""

# FLOW
_C.FLOW = CN()
_C.FLOW.N_FLOW = 8
_C.FLOW.N_BLOCK = 1
_C.FLOW.IN_FEAT = 2
_C.FLOW.MLP_DIM = 23
_C.FLOW.N_BITS = 5
_C.FLOW.N_BINS = 32
_C.FLOW.DROPOUT = False
_C.FLOW.INIT_ZEROS = False

# DATASET
_C.DATASET = CN()
_C.DATASET.IN_DIST = "BU3D"
_C.DATASET.N_CLASS = 2
_C.DATASET.IMG_SIZE=224
_C.DATASET.NUM_WORKERS=1
_C.DATASET.AUG = False
_C.DATASET.W_SAMPLER = False

# LOSS
_C.LOSS = CN()
_C.LOSS.LMBDA_MIN = 0.07
_C.LOSS.LMBDA_MAX = 0.0051
_C.LOSS.CYCLE = 50
_C.LOSS.TAU = 1.5
_C.LOSS.TAU2 = 0.1

# TRAINING
_C.TRAINING = CN()
_C.TRAINING.ITER = 1000
_C.TRAINING.BATCH = 256
_C.TRAINING.LR = 1e-3
_C.TRAINING.WT_DECAY = 1e-5
_C.TRAINING.MOMENTUM = 0.9
_C.TRAINING.DROPOUT = 0
_C.TRAINING.PRETRAINED = "dense"
_C.TRAINING.PRT_CONFIG = 1
_C.TRAINING.PRT_LAYER = 1

# LEARNING RATE
_C.LR = CN()
_C.LR.WARM = False
_C.LR.ADJUST = False
_C.LR.WARM_ITER = 10
_C.LR.WARMUP_FROM = 1e-5
_C.LR.DECAY_RATE = 0.1
_C.LR.MIN_LR = 1e-6
_C.LR.T_MAX = 100

# TEST
_C.TEST = CN()
_C.TEST.EMP_PARAMS = False
_C.TEST.SCORE = False
_C.TEST.MAGNITUDE = 0.001
_C.TEST.IN_FEATS = []

# COMMENTS
_C.COMMENTS = "TEST"




def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()