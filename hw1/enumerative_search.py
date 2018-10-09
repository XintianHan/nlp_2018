from os import listdir
import spacy
import string
import pickle as pkl
import numpy as np
from nltk import ngrams
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
%matplotlib inline
from collections import Counter

