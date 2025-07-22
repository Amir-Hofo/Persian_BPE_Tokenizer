import os, re
from zipfile import ZipFile

import requests
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from hazm import Normalizer