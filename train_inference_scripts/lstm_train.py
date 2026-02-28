"""
TERRATRENDS LSTM TRAINER v2
============================
Improvements over v1:
  1. Target = 3-year rolling average growth (smoother, more learnable)
  2. Cross-county neighbor features (spillover signals)
  3. Year embeddings (business cycle awareness)
  4. Classification target: shrinking / flat / moderate / strong
     → easier ML problem, maps cleanly to scoring
  5. 2020 excluded (COVID structural break)
  6. Class-weighted loss to handle imbalance

Architecture:
  - Sector embedding   (20 sectors   → 8-dim)
  - County embedding   (159 counties → 16-dim)
  - Year embedding     (21 years     → 4-dim)
  - Economic features  (5 macro features, normalized)
  - Neighbor features  (avg growth of neighboring counties, same sector)
  - LSTM (128 hidden, 2 layers, dropout=0.3)
  - Linear head → 4-class softmax per forecast step

Class definitions (on 3yr rolling avg growth rate):
  0  shrinking  : < -5%
  1  flat       : -5% to +5%
  2  moderate   :  +5% to +20%
  3  strong     : > +20%

Output:
  lstm_model_v2.pt  — model weights + full config
  lstm_eval_v2.csv  — per-sector classification metrics
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SEED             = 42
SEQ_LEN          = 10
FORECAST_HORIZON = 3       # predict 3 steps: 1Y, 2Y, 3Y ahead
                            # (from which we derive 1Y/3Y/5Y scores)
N_CLASSES        = 4
BATCH_SIZE       = 512
EPOCHS           = 150
LR               = 3e-4
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.3
SECTOR_EMB_DIM   = 8
COUNTY_EMB_DIM   = 16
YEAR_EMB_DIM     = 4
WEIGHT_DECAY     = 1e-4
N_NEIGHBORS      = 3       # top-N neighbors to include as features

# Class bins (applied to 3yr rolling average growth rate)
CLASS_BINS   = [-np.inf, -0.05, 0.05, 0.20, np.inf]
CLASS_LABELS = ["shrinking", "flat", "moderate", "strong"]

# Growth rate multipliers for revenue projection (midpoint of each class)
CLASS_GROWTH_RATES = {
    0: -0.12,   # shrinking: ~-12% per year
    1:  0.00,   # flat: ~0%
    2:  0.10,   # moderate: ~+10% per year
    3:  0.28,   # strong: ~+28% per year
}

MACRO_FEATURES = [
    "Unemployment_Rate",
    "Per_Capita_Personal_Income",
    "Real_GDP",
    "Percent_Change_Real_GDP",
    "Bachelor_Degree_or_Higher_Pct",
]

SECTOR_COLS = [
    'Accommodation and food services',
    'Administrative and support and waste management and remediation services',
    'Agriculture, forestry, fishing and hunting',
    'Arts, entertainment, and recreation',
    'Construction',
    'Durable goods manufacturing',
    'Educational services',
    'Finance and insurance',
    'Government and government enterprises',
    'Health care and social assistance',
    'Information',
    'Natural resources and mining',
    'Nondurable goods manufacturing',
    'Other services (except government and government enterprises)',
    'Professional and business services',
    'Real estate and rental and leasing',
    'Retail trade',
    'Transportation and warehousing',
    'Utilities',
    'Wholesale trade'
]

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# -------------------------------------------------------------------
# Georgia neighbor map (FIPS → list of neighbor FIPS)
# -------------------------------------------------------------------
GA_NEIGHBORS = {
    13001: [13005, 13025, 13127, 13169, 13299],
    13003: [13001, 13127, 13183, 13299],
    13005: [13001, 13003, 13025, 13127],
    13007: [13037, 13095, 13155, 13193, 13273],
    13009: [13011, 13057, 13113, 13121, 13137, 13195],
    13011: [13009, 13085, 13113, 13137, 13139],
    13013: [13057, 13085, 13117, 13121, 13139],
    13015: [13057, 13067, 13085, 13117, 13223, 13233],
    13017: [13075, 13101, 13173, 13187, 13321],
    13019: [13075, 13101, 13173, 13185, 13299],
    13021: [13049, 13077, 13079, 13153, 13169, 13207],
    13023: [13021, 13077, 13153, 13207, 13303],
    13025: [13001, 13003, 13005, 13127, 13161, 13299],
    13027: [13019, 13075, 13185, 13261, 13275],
    13029: [13031, 13051, 13103, 13179],
    13031: [13029, 13051, 13067, 13103, 13179, 13245],
    13033: [13009, 13049, 13121, 13153, 13245, 13279],
    13035: [13049, 13077, 13113, 13199, 13207],
    13037: [13007, 13095, 13193, 13273, 13309],
    13039: [13025, 13127, 13161],
    13041: [13045, 13057, 13067, 13097, 13143, 13149, 13231, 13233],
    13043: [13057, 13085, 13115, 13117, 13139, 13211],
    13045: [13041, 13143, 13149, 13213, 13231],
    13047: [13021, 13049, 13079, 13153, 13197, 13255],
    13049: [13021, 13033, 13035, 13077, 13113, 13153, 13199, 13207],
    13051: [13029, 13031, 13103, 13163, 13179, 13245, 13265],
    13053: [13067, 13085, 13117, 13223, 13281, 13295],
    13055: [13085, 13139, 13187, 13281],
    13057: [13009, 13013, 13015, 13041, 13043, 13085, 13113, 13117, 13121, 13139],
    13059: [13007, 13037, 13095, 13273, 13309],
    13061: [13021, 13049, 13077, 13079, 13169, 13207],
    13063: [13029, 13031, 13051, 13245, 13265],
    13065: [13047, 13079, 13143, 13149, 13197, 13227, 13285],
    13067: [13015, 13031, 13041, 13053, 13057, 13085, 13117, 13223],
    13069: [13021, 13077, 13169, 13207, 13255],
    13071: [13041, 13097, 13143, 13285],
    13073: [13009, 13113, 13121, 13195, 13279],
    13075: [13017, 13019, 13027, 13101, 13173, 13185, 13261, 13275],
    13077: [13021, 13023, 13035, 13049, 13061, 13069, 13153, 13169, 13207],
    13079: [13021, 13047, 13061, 13065, 13149, 13197, 13227, 13255],
    13081: [13055, 13085, 13139, 13187, 13223, 13281],
    13083: [13093, 13131, 13141, 13171, 13243, 13291],
    13085: [13011, 13013, 13015, 13043, 13053, 13055, 13057, 13067, 13081, 13115, 13117, 13139, 13187, 13223, 13281],
    13087: [13049, 13121, 13195, 13245, 13279],
    13089: [13061, 13065, 13077, 13149, 13151, 13197, 13227, 13247],
    13091: [13021, 13049, 13077, 13153, 13199, 13207, 13303],
    13093: [13007, 13037, 13059, 13095, 13155, 13193, 13273, 13309],
    13095: [13007, 13037, 13059, 13093, 13155, 13193, 13273],
    13097: [13041, 13071, 13143, 13149, 13213, 13231, 13285],
    13099: [13009, 13057, 13113, 13121, 13137, 13195],
    13101: [13017, 13075, 13173, 13185, 13187, 13261, 13321],
    13103: [13029, 13031, 13051, 13163, 13245, 13265, 13267],
    13105: [13009, 13073, 13113, 13121, 13137, 13195, 13279],
    13107: [13043, 13085, 13115, 13117, 13139, 13211, 13281],
    13109: [13075, 13101, 13173, 13187, 13261, 13321],
    13111: [13021, 13049, 13079, 13153, 13169, 13255],
    13113: [13009, 13011, 13033, 13035, 13049, 13057, 13073, 13099, 13105, 13121, 13195],
    13115: [13043, 13085, 13107, 13117, 13211, 13281],
    13117: [13013, 13015, 13043, 13053, 13057, 13067, 13085, 13107, 13115, 13223, 13281],
    13119: [13007, 13155, 13193, 13273],
    13121: [13009, 13011, 13013, 13033, 13049, 13057, 13073, 13099, 13105, 13113, 13195, 13279],
    13123: [13007, 13059, 13095, 13155, 13193, 13309],
    13125: [13051, 13087, 13163, 13189, 13237, 13265, 13289],
    13127: [13001, 13003, 13005, 13025, 13039, 13161, 13299],
    13129: [13027, 13075, 13185, 13275, 13299],
    13131: [13083, 13093, 13141, 13171, 13193, 13243],
    13133: [13029, 13031, 13051, 13103, 13179, 13245, 13265],
    13135: [13019, 13027, 13075, 13185, 13275, 13299],
    13137: [13009, 13011, 13099, 13105, 13121, 13195],
    13139: [13011, 13013, 13043, 13055, 13057, 13081, 13085, 13107, 13113, 13187, 13281],
    13141: [13083, 13093, 13131, 13155, 13171, 13193, 13243, 13273],
    13143: [13041, 13045, 13065, 13071, 13079, 13097, 13149, 13197, 13213, 13231, 13285],
    13145: [13047, 13079, 13149, 13197, 13215, 13227, 13255, 13285],
    13147: [13051, 13087, 13125, 13163, 13189, 13237, 13265],
    13149: [13041, 13045, 13065, 13071, 13079, 13089, 13097, 13143, 13145, 13197, 13213, 13227, 13231, 13247, 13285],
    13151: [13061, 13065, 13079, 13089, 13149, 13197, 13227, 13247],
    13153: [13021, 13023, 13033, 13035, 13049, 13061, 13069, 13077, 13091, 13207, 13303],
    13155: [13007, 13037, 13059, 13093, 13095, 13119, 13123, 13141, 13193, 13273],
    13157: [13047, 13069, 13079, 13145, 13215, 13255, 13285],
    13159: [13051, 13087, 13125, 13147, 13163, 13189, 13237, 13265, 13289],
    13161: [13025, 13039, 13127, 13299],
    13163: [13029, 13031, 13051, 13087, 13103, 13125, 13133, 13147, 13179, 13189, 13237, 13245, 13265, 13289],
    13165: [13047, 13069, 13145, 13157, 13197, 13215, 13255, 13285],
    13167: [13043, 13085, 13107, 13115, 13117, 13139, 13187, 13223, 13281],
    13169: [13001, 13003, 13021, 13023, 13061, 13069, 13077, 13091, 13153, 13207, 13303],
    13171: [13083, 13093, 13131, 13141, 13155, 13193, 13243, 13273, 13309],
    13173: [13017, 13075, 13101, 13109, 13185, 13261, 13321],
    13175: [13047, 13069, 13145, 13157, 13197, 13215, 13255],
    13177: [13009, 13073, 13105, 13113, 13121, 13137, 13195, 13279],
    13179: [13029, 13031, 13051, 13103, 13133, 13163, 13245, 13265],
    13181: [13047, 13069, 13079, 13145, 13157, 13175, 13197, 13215, 13255],
    13183: [13001, 13003, 13127, 13169, 13299],
    13185: [13019, 13027, 13075, 13101, 13129, 13135, 13173, 13261, 13275, 13299, 13321],
    13187: [13055, 13081, 13085, 13101, 13109, 13115, 13117, 13139, 13167, 13223, 13281],
    13189: [13051, 13087, 13125, 13147, 13159, 13163, 13237, 13265, 13289],
    13191: [13047, 13069, 13145, 13165, 13175, 13197, 13215, 13255, 13285],
    13193: [13007, 13037, 13059, 13083, 13093, 13095, 13119, 13123, 13131, 13141, 13155, 13171, 13273, 13309],
    13195: [13009, 13073, 13099, 13105, 13113, 13121, 13137, 13177, 13279],
    13197: [13047, 13061, 13065, 13079, 13089, 13145, 13149, 13151, 13157, 13165, 13175, 13181, 13191, 13215, 13227, 13247, 13255, 13285],
    13199: [13021, 13035, 13049, 13077, 13091, 13153, 13207, 13303],
    13201: [13051, 13087, 13125, 13147, 13163, 13189, 13237, 13265, 13289],
    13203: [13009, 13073, 13099, 13105, 13121, 13137, 13195, 13279],
    13205: [13049, 13077, 13091, 13153, 13169, 13199, 13207, 13303],
    13207: [13021, 13023, 13035, 13049, 13061, 13069, 13077, 13091, 13111, 13153, 13169, 13199, 13205, 13255, 13303],
    13209: [13051, 13087, 13125, 13163, 13189, 13201, 13237, 13265, 13289],
    13211: [13043, 13085, 13107, 13115, 13117, 13139, 13167, 13187, 13223, 13281],
    13213: [13041, 13045, 13071, 13097, 13143, 13149, 13231, 13285],
    13215: [13047, 13079, 13145, 13157, 13165, 13175, 13181, 13191, 13197, 13255, 13285],
    13217: [13051, 13087, 13125, 13163, 13189, 13201, 13209, 13237, 13265, 13289],
    13219: [13009, 13073, 13099, 13105, 13121, 13137, 13177, 13195, 13203, 13279],
    13221: [13027, 13075, 13129, 13135, 13173, 13185, 13261, 13275, 13321],
    13223: [13015, 13053, 13055, 13067, 13081, 13085, 13117, 13139, 13187, 13211, 13281],
    13225: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13207, 13303],
    13227: [13047, 13061, 13065, 13079, 13089, 13145, 13149, 13151, 13197, 13215, 13247, 13255, 13285],
    13229: [13001, 13003, 13025, 13127, 13161, 13169, 13183, 13299],
    13231: [13041, 13045, 13071, 13097, 13143, 13149, 13213, 13285],
    13233: [13015, 13041, 13067, 13117, 13143, 13223],
    13235: [13027, 13075, 13129, 13135, 13173, 13185, 13221, 13261, 13275, 13321],
    13237: [13051, 13087, 13125, 13147, 13159, 13163, 13189, 13201, 13209, 13265, 13289],
    13239: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13279],
    13241: [13019, 13027, 13075, 13129, 13135, 13185, 13221, 13261, 13275, 13299, 13321],
    13243: [13083, 13093, 13131, 13141, 13155, 13171, 13193, 13273, 13309],
    13245: [13029, 13031, 13033, 13051, 13063, 13087, 13103, 13125, 13133, 13163, 13179, 13189, 13265],
    13247: [13061, 13065, 13089, 13149, 13151, 13197, 13227, 13255],
    13249: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13303],
    13251: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13279],
    13253: [13051, 13087, 13125, 13163, 13189, 13201, 13209, 13217, 13237, 13265, 13289],
    13255: [13021, 13047, 13061, 13069, 13079, 13089, 13111, 13145, 13149, 13151, 13157, 13165, 13175, 13181, 13191, 13197, 13207, 13215, 13227, 13247, 13285],
    13257: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13279],
    13259: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13303],
    13261: [13017, 13075, 13101, 13109, 13173, 13185, 13221, 13235, 13275, 13321],
    13263: [13051, 13087, 13103, 13125, 13133, 13163, 13179, 13189, 13245, 13265],
    13265: [13029, 13031, 13033, 13051, 13063, 13087, 13103, 13125, 13133, 13147, 13163, 13179, 13189, 13201, 13209, 13217, 13237, 13245, 13253, 13263, 13289],
    13267: [13029, 13031, 13051, 13063, 13103, 13133, 13179, 13245, 13265],
    13269: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13257, 13279],
    13271: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13303],
    13273: [13007, 13037, 13059, 13083, 13093, 13095, 13119, 13123, 13131, 13141, 13155, 13171, 13193, 13243, 13309],
    13275: [13019, 13027, 13075, 13101, 13129, 13135, 13173, 13185, 13221, 13235, 13241, 13261, 13321],
    13277: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13303],
    13279: [13009, 13033, 13073, 13099, 13105, 13113, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13257, 13269],
    13281: [13015, 13043, 13053, 13055, 13067, 13081, 13085, 13107, 13115, 13117, 13139, 13167, 13187, 13211, 13223],
    13283: [13051, 13087, 13103, 13125, 13133, 13163, 13179, 13189, 13245, 13263, 13265, 13267],
    13285: [13041, 13045, 13065, 13071, 13079, 13089, 13097, 13143, 13145, 13149, 13151, 13157, 13165, 13175, 13181, 13191, 13197, 13213, 13215, 13227, 13231, 13247, 13255],
    13287: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13257, 13269, 13279],
    13289: [13051, 13087, 13125, 13147, 13159, 13163, 13189, 13201, 13209, 13217, 13237, 13245, 13253, 13265],
    13291: [13083, 13093, 13131, 13141, 13155, 13171, 13193, 13243, 13273, 13309],
    13293: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13277, 13303],
    13295: [13015, 13053, 13055, 13067, 13081, 13085, 13115, 13117, 13139, 13167, 13187, 13211, 13223, 13281],
    13297: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13277, 13293, 13303],
    13299: [13001, 13003, 13005, 13019, 13025, 13027, 13039, 13075, 13101, 13127, 13129, 13135, 13161, 13183, 13185, 13229, 13241],
    13301: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13257, 13269, 13279, 13287],
    13303: [13021, 13023, 13049, 13077, 13091, 13153, 13169, 13199, 13205, 13207, 13225, 13249, 13259, 13271, 13277, 13293, 13297],
    13305: [13051, 13087, 13103, 13125, 13133, 13163, 13179, 13189, 13245, 13263, 13265, 13267, 13283],
    13307: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13277, 13293, 13297, 13303],
    13309: [13037, 13059, 13083, 13093, 13095, 13119, 13123, 13131, 13141, 13155, 13171, 13193, 13243, 13273, 13291],
    13311: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13277, 13293, 13297, 13303, 13307],
    13313: [13015, 13043, 13053, 13055, 13067, 13081, 13085, 13107, 13115, 13117, 13139, 13167, 13187, 13211, 13223, 13281, 13295],
    13315: [13009, 13073, 13105, 13121, 13137, 13177, 13195, 13203, 13219, 13239, 13251, 13257, 13269, 13279, 13287, 13301],
    13317: [13049, 13077, 13091, 13153, 13169, 13199, 13205, 13225, 13249, 13259, 13271, 13277, 13293, 13297, 13303, 13307, 13311],
    13319: [13051, 13087, 13103, 13125, 13133, 13163, 13179, 13189, 13245, 13263, 13265, 13267, 13283, 13305],
    13321: [13017, 13075, 13101, 13109, 13173, 13185, 13221, 13235, 13241, 13261, 13275],
}

# -------------------------------------------------------------------
# Load & preprocess
# -------------------------------------------------------------------
print("\nLoading data...")
df = pd.read_csv("data/merged_data.csv").sort_values(["County", "Year"])
df = df[df["Year"] != 2020].reset_index(drop=True)

counties   = sorted(df["County"].unique())
fips_list  = sorted(df["GeoID"].unique())
county2idx = {c: i for i, c in enumerate(counties)}
fips2idx   = {f: i for i, f in enumerate(fips_list)}
fips2county = dict(zip(df["GeoID"], df["County"]))
county2fips = {v: k for k, v in fips2county.items()}

years     = sorted(df["Year"].unique())
year2idx  = {y: i for i, y in enumerate(years)}
N_YEARS   = len(years)
N_COUNTIES = len(counties)
N_SECTORS  = len(SECTOR_COLS)

print(f"✓ {N_COUNTIES} counties, {N_SECTORS} sectors, {N_YEARS} years (2020 excluded)")

# -------------------------------------------------------------------
# Compute 3-year rolling average growth rates (the smoothed target)
# -------------------------------------------------------------------
print("\nComputing 3yr rolling averages...")
rolling_df = df.copy()
for sector in SECTOR_COLS:
    for county in counties:
        mask = rolling_df["County"] == county
        rolling_df.loc[mask, sector] = (
            rolling_df.loc[mask, sector]
            .rolling(3, min_periods=2)
            .mean()
        )

# -------------------------------------------------------------------
# Per-sector scalers (fit on raw values, used for neighbor features)
# -------------------------------------------------------------------
print("Fitting scalers...")
sector_scalers = {}
for sector in SECTOR_COLS:
    vals = df[sector].dropna().values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(vals)
    sector_scalers[sector] = scaler

macro_scalers = {}
for feat in MACRO_FEATURES:
    vals = df[feat].dropna().values.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(vals)
    macro_scalers[feat] = scaler

# -------------------------------------------------------------------
# Imputation
# -------------------------------------------------------------------
print("Imputing...")
df_imp       = df.copy()
rolling_imp  = rolling_df.copy()

for sector in SECTOR_COLS:
    for county in counties:
        mask = df_imp["County"] == county
        df_imp.loc[mask, sector]      = df_imp.loc[mask, sector].ffill().bfill()
        rolling_imp.loc[mask, sector] = rolling_imp.loc[mask, sector].ffill().bfill()
    df_imp[sector]      = df_imp[sector].fillna(df_imp[sector].mean())
    rolling_imp[sector] = rolling_imp[sector].fillna(rolling_imp[sector].mean())

for feat in MACRO_FEATURES:
    for county in counties:
        mask = df_imp["County"] == county
        df_imp.loc[mask, feat] = df_imp.loc[mask, feat].ffill().bfill()
    df_imp[feat] = df_imp[feat].fillna(df_imp[feat].mean())

# -------------------------------------------------------------------
# Build arrays
# sector_arr    [N_COUNTIES, N_YEARS, N_SECTORS] — normalized raw growth rates
# rolling_arr   [N_COUNTIES, N_YEARS, N_SECTORS] — normalized 3yr rolling avg
# macro_arr     [N_COUNTIES, N_YEARS, N_MACRO]
# label_arr     [N_COUNTIES, N_YEARS, N_SECTORS] — class labels 0-3
# -------------------------------------------------------------------
print("Building arrays...")
sector_arr  = np.zeros((N_COUNTIES, N_YEARS, N_SECTORS), dtype=np.float32)
rolling_arr = np.zeros((N_COUNTIES, N_YEARS, N_SECTORS), dtype=np.float32)
macro_arr   = np.zeros((N_COUNTIES, N_YEARS, len(MACRO_FEATURES)), dtype=np.float32)
label_arr   = np.zeros((N_COUNTIES, N_YEARS, N_SECTORS), dtype=np.int64)

for county in counties:
    ci      = county2idx[county]
    cdf     = df_imp[df_imp["County"] == county].sort_values("Year")
    crdf    = rolling_imp[rolling_imp["County"] == county].sort_values("Year")

    for year in years:
        yi   = year2idx[year]
        row  = cdf[cdf["Year"] == year]
        rrow = crdf[crdf["Year"] == year]
        if row.empty:
            continue

        for si, sector in enumerate(SECTOR_COLS):
            raw = row[sector].values[0]
            rol = rrow[sector].values[0]
            sector_arr[ci, yi, si]  = sector_scalers[sector].transform([[raw]])[0][0]
            rolling_arr[ci, yi, si] = sector_scalers[sector].transform([[rol]])[0][0]

            # Class label from rolling avg (denormalized)
            label_arr[ci, yi, si] = np.searchsorted(
                CLASS_BINS[1:-1], rol
            )

        for fi, feat in enumerate(MACRO_FEATURES):
            raw = row[feat].values[0]
            macro_arr[ci, yi, fi] = macro_scalers[feat].transform([[raw]])[0][0]

print(f"✓ Arrays: sector={sector_arr.shape}, labels={label_arr.shape}")

# -------------------------------------------------------------------
# Neighbor feature lookup
# avg normalized growth rate of N_NEIGHBORS nearest counties, same sector
# -------------------------------------------------------------------
# Build FIPS → county_idx map
fips2ci = {}
for county in counties:
    fips = county2fips.get(county)
    if fips:
        fips2ci[fips] = county2idx[county]

def get_neighbor_avg(ci, si, yi, n=N_NEIGHBORS):
    county  = counties[ci]
    fips    = county2fips.get(county)
    if not fips:
        return 0.0
    neighbor_fips = GA_NEIGHBORS.get(fips, [])
    vals = []
    for nf in neighbor_fips[:n]:
        nci = fips2ci.get(nf)
        if nci is not None:
            vals.append(sector_arr[nci, yi, si])
    return float(np.mean(vals)) if vals else 0.0

# -------------------------------------------------------------------
# Class weights for imbalanced loss
# -------------------------------------------------------------------
all_labels = label_arr.flatten()
class_counts = np.bincount(all_labels, minlength=N_CLASSES).astype(float)
class_weights = torch.tensor(
    1.0 / (class_counts / class_counts.sum()),
    dtype=torch.float32
)
class_weights = class_weights / class_weights.sum() * N_CLASSES
print(f"\nClass distribution: {dict(zip(CLASS_LABELS, class_counts.astype(int)))}")
print(f"Class weights: {class_weights.numpy().round(3)}")

# -------------------------------------------------------------------
# Dataset
# Input per timestep:
#   macro (5) + sector_rate (1) + neighbor_avg (1) + imputed_mask (1) = 8
# -------------------------------------------------------------------
INPUT_SIZE = len(MACRO_FEATURES) + 1 + 1 + 1  # macro + sector + neighbor + mask

class TerraDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ci, si, yi_start, x_seq, y_labels = self.samples[idx]
        return (
            torch.tensor(ci,       dtype=torch.long),
            torch.tensor(si,       dtype=torch.long),
            torch.tensor(yi_start, dtype=torch.long),
            torch.tensor(x_seq,    dtype=torch.float32),
            torch.tensor(y_labels, dtype=torch.long),
        )


def build_samples(split="train"):
    samples   = []
    max_start = N_YEARS - SEQ_LEN - FORECAST_HORIZON

    for ci in range(N_COUNTIES):
        for si in range(N_SECTORS):
            for t in range(max_start + 1):
                target_start = t + SEQ_LEN

                if split == "train" and target_start > 13: continue
                if split == "val"   and target_start != 14: continue
                if split == "test"  and target_start <= 14: continue

                # Build input sequence
                x_seq = []
                for ti in range(t, t + SEQ_LEN):
                    macro_vec    = macro_arr[ci, ti, :].tolist()
                    sector_val   = float(sector_arr[ci, ti, si])
                    neighbor_val = get_neighbor_avg(ci, si, ti)
                    imputed_mask = 0.0
                    x_seq.append(macro_vec + [sector_val, neighbor_val, imputed_mask])

                # Target: class labels for next FORECAST_HORIZON steps
                y_labels = label_arr[ci, target_start:target_start + FORECAST_HORIZON, si]
                if len(y_labels) < FORECAST_HORIZON:
                    continue

                samples.append((ci, si, t, np.array(x_seq, dtype=np.float32), y_labels))

    return samples


print("\nBuilding datasets (this takes ~1 min due to neighbor lookups)...")
train_samples = build_samples("train")
val_samples   = build_samples("val")
test_samples  = build_samples("test")

print(f"  Train: {len(train_samples):,}")
print(f"  Val:   {len(val_samples):,}")
print(f"  Test:  {len(test_samples):,}")

if len(val_samples) == 0 or len(test_samples) == 0:
    raise ValueError("Empty split — check SEQ_LEN + FORECAST_HORIZON vs N_YEARS")

train_loader = DataLoader(TerraDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TerraDataset(val_samples),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(TerraDataset(test_samples),  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------
class TerraLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sector_emb = nn.Embedding(N_SECTORS,  SECTOR_EMB_DIM)
        self.county_emb = nn.Embedding(N_COUNTIES, COUNTY_EMB_DIM)
        self.year_emb   = nn.Embedding(N_YEARS,    YEAR_EMB_DIM)

        proj_input = INPUT_SIZE + SECTOR_EMB_DIM + COUNTY_EMB_DIM + YEAR_EMB_DIM
        self.input_proj = nn.Linear(proj_input, HIDDEN_SIZE)

        self.lstm = nn.LSTM(
            input_size=HIDDEN_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )
        self.dropout = nn.Dropout(DROPOUT)

        # One classification head per forecast step
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, N_CLASSES)
            )
            for _ in range(FORECAST_HORIZON)
        ])

    def forward(self, county_idx, sector_idx, year_idx, x_seq):
        B, T, _ = x_seq.shape

        s_emb = self.sector_emb(sector_idx).unsqueeze(1).expand(B, T, -1)
        c_emb = self.county_emb(county_idx).unsqueeze(1).expand(B, T, -1)

        # Year embedding: one per timestep in sequence
        yr_range = year_idx.unsqueeze(1) + torch.arange(T, device=x_seq.device).unsqueeze(0)
        yr_range = yr_range.clamp(0, N_YEARS - 1)
        y_emb    = self.year_emb(yr_range)   # [B, T, YEAR_EMB_DIM]

        x = torch.cat([x_seq, s_emb, c_emb, y_emb], dim=-1)
        x = torch.relu(self.input_proj(x))

        out, _  = self.lstm(x)
        last    = self.dropout(out[:, -1, :])

        # One logit vector per forecast step
        logits = [head(last) for head in self.heads]  # list of [B, N_CLASSES]
        return torch.stack(logits, dim=1)              # [B, FORECAST_HORIZON, N_CLASSES]


model     = TerraLSTM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

total_params = sum(p.numel() for p in model.parameters())
print(f"\n✓ Model: {total_params:,} parameters")

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
print("\n" + "="*70)
print("TRAINING")
print("="*70)

best_val_loss = float("inf")
best_state    = None
patience      = 20
patience_ctr  = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for ci, si, yi, x_seq, y_labels in train_loader:
        ci, si, yi = ci.to(device), si.to(device), yi.to(device)
        x_seq      = x_seq.to(device)
        y_labels   = y_labels.to(device)   # [B, FORECAST_HORIZON]

        optimizer.zero_grad()
        logits = model(ci, si, yi, x_seq)  # [B, FORECAST_HORIZON, N_CLASSES]

        # Compute cross-entropy over all forecast steps
        loss = sum(
            criterion(logits[:, step, :], y_labels[:, step])
            for step in range(FORECAST_HORIZON)
        ) / FORECAST_HORIZON

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ci, si, yi, x_seq, y_labels in val_loader:
            ci, si, yi = ci.to(device), si.to(device), yi.to(device)
            logits     = model(ci, si, yi, x_seq.to(device))
            y_labels   = y_labels.to(device)
            val_loss  += sum(
                criterion(logits[:, step, :], y_labels[:, step])
                for step in range(FORECAST_HORIZON)
            ).item() / FORECAST_HORIZON
    val_loss /= len(val_loader)
    scheduler.step()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_ctr  = 0
    else:
        patience_ctr += 1

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}  best={best_val_loss:.4f}  patience={patience_ctr}/{patience}")

    if patience_ctr >= patience:
        print(f"\n  Early stopping at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"\n✓ Best val loss: {best_val_loss:.4f}")

# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
print("\n" + "="*70)
print("EVALUATION (test set — 2017-2023 excl. 2020)")
print("="*70)

model.eval()
sector_true = {s: [] for s in SECTOR_COLS}
sector_pred = {s: [] for s in SECTOR_COLS}

with torch.no_grad():
    for ci, si, yi, x_seq, y_labels in test_loader:
        logits = model(
            ci.to(device), si.to(device), yi.to(device), x_seq.to(device)
        ).cpu()
        preds    = logits.argmax(dim=-1)   # [B, FORECAST_HORIZON]
        s_idxs   = si.numpy()
        y_true   = y_labels.numpy()
        y_pred   = preds.numpy()

        for b in range(len(s_idxs)):
            sector = SECTOR_COLS[s_idxs[b]]
            sector_true[sector].extend(y_true[b].tolist())
            sector_pred[sector].extend(y_pred[b].tolist())

eval_results = {}
all_true, all_pred = [], []

for sector in SECTOR_COLS:
    t = sector_true[sector]
    p = sector_pred[sector]
    if len(t) < 4:
        continue
    f1  = f1_score(t, p, average="macro", zero_division=0)
    acc = np.mean(np.array(t) == np.array(p))
    eval_results[sector] = {"F1_macro": f1, "Accuracy": acc, "N": len(t)}
    all_true.extend(t)
    all_pred.extend(p)
    print(f"  {sector[:48]:<48}  F1={f1:.3f}  Acc={acc:.3f}")

overall_f1  = f1_score(all_true, all_pred, average="macro", zero_division=0)
overall_acc = np.mean(np.array(all_true) == np.array(all_pred))
print(f"\n  Overall  F1={overall_f1:.3f}  Acc={overall_acc:.3f}")
print(f"\n  Confusion matrix (all sectors):")
print(confusion_matrix(all_true, all_pred))
print(f"  Classes: {CLASS_LABELS}")

eval_df = pd.DataFrame(eval_results).T.sort_values("F1_macro", ascending=False)
eval_df.to_csv("lstm_eval_v2.csv")
print(f"\n✓ Eval saved to lstm_eval_v2.csv")

# -------------------------------------------------------------------
# Save
# -------------------------------------------------------------------
print("\nSaving...")
torch.save({
    "model_state":       best_state,
    "config": {
        "hidden_size":       HIDDEN_SIZE,
        "num_layers":        NUM_LAYERS,
        "dropout":           DROPOUT,
        "sector_emb_dim":    SECTOR_EMB_DIM,
        "county_emb_dim":    COUNTY_EMB_DIM,
        "year_emb_dim":      YEAR_EMB_DIM,
        "input_size":        INPUT_SIZE,
        "forecast_horizon":  FORECAST_HORIZON,
        "seq_len":           SEQ_LEN,
        "n_sectors":         N_SECTORS,
        "n_counties":        N_COUNTIES,
        "n_years":           N_YEARS,
        "n_classes":         N_CLASSES,
    },
    "sector_cols":       SECTOR_COLS,
    "macro_features":    MACRO_FEATURES,
    "county2idx":        county2idx,
    "sector2idx":        {s: i for i, s in enumerate(SECTOR_COLS)},
    "year2idx":          year2idx,
    "county2fips":       county2fips,
    "ga_neighbors":      GA_NEIGHBORS,
    "sector_scalers":    sector_scalers,
    "macro_scalers":     macro_scalers,
    "class_bins":        CLASS_BINS,
    "class_labels":      CLASS_LABELS,
    "class_growth_rates": CLASS_GROWTH_RATES,
    "years":             years,
    "model_version":     "terratrends_lstm_v2",
}, "lstm_model_v2.pt")

print("✓ Saved to lstm_model_v2.pt")
print("="*70 + "\n")