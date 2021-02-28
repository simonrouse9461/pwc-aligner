import sys
import math
import random
import warnings
from functools import *
from typing import *

import numpy as np
import pandas as pd
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import metrics
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from transformers import (AutoTokenizer,
                          AutoModel, 
                          AutoModelForMaskedLM, 
                          AutoModelForNextSentencePrediction, 
                          AutoModelForSequenceClassification,
                          PreTrainedTokenizer,
                          PreTrainedModel,
                          AdamW)