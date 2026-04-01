import os

# Root paths
PATH_CORPORA = os.path.expanduser("~/lltk_data/corpora")
PATH_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PATH_STASH = os.path.join(PATH_DATA, "stash")
PATH_MODELS = os.path.join(PATH_DATA, "models")
PATH_FIGS = os.path.join(PATH_DATA, "figures")

# Word norms / fields
FIELD_DIR = os.path.join(PATH_DATA, "fields")
SOURCE_DIR = os.path.join(FIELD_DIR, "sources")
COUNT_DIR = os.path.join(PATH_DATA, "counts")
PSGS_DIR = os.path.join(PATH_DATA, "psgs")
SCORES_DIR = os.path.join(PATH_DATA, "scores")
DIST_DIR = os.path.join(PATH_DATA, "dists")

PATH_NORMS = os.path.join(FIELD_DIR, "data.wordnorms_orig.csv")
PATH_ALLNORMS = os.path.join(FIELD_DIR, "data.allnorms.pkl.gz")
PATH_VECNORMS = os.path.join(FIELD_DIR, "data.wordnorms_vec.csv.gz")
PATH_VECFIELDS = os.path.join(FIELD_DIR, "data.fields_vec.csv.gz")
PATH_STOPWORDS = os.path.join(FIELD_DIR, "stopwords.txt")
PATH_NAMES = os.path.join(FIELD_DIR, "capslocked.CanonFiction.txt")
PATH_SPELLING_D = os.path.join(FIELD_DIR, "spelling_variants_from_morphadorner.txt")

# Z-score cutoff for classifying words as abstract/concrete
ZCUT = 1.0

# Norm sources
SOURCES_FOR_COUNTING = {"Median"}
SOURCES_FOR_PLOTTING = {"PAV-Conc", "PAV-Imag", "MRC-Conc", "MRC-Imag", "MT-Conc", "LSN-Imag", "Median"}
BAD_SOURCES = {"LSN-Perc", "LSN-Sens"}

# Counting
COUNT_WINDOW_LEN = 100
REMOVE_STOPWORDS = True
MODERNIZE_SPELLING = False

# Word2Vec model training
MODEL_PERIOD_LEN = 100
MODEL_MIN_COUNT = 10
MODEL_NUM_DIM = 100
