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
PATH_VECNORMS = os.path.join(FIELD_DIR, "data.wordnorms_vec.pkl.gz")
PATH_VECFIELDS = os.path.join(FIELD_DIR, "data.fields_vec.csv.gz")

PATH_NORMS_FR = os.path.join(FIELD_DIR, "data.wordnorms_orig.fr.csv")
PATH_VECNORMS_FR = os.path.join(FIELD_DIR, "data.wordnorms_vec.fr.pkl.gz")
PATH_ALLNORMS_FR = os.path.join(FIELD_DIR, "data.allnorms.fr.pkl.gz")
PATH_MODELS_FR = os.path.join(PATH_DATA, "models_fr")
FR_SOURCE_DIR = os.path.join(SOURCE_DIR, "fr_norms")
PATH_BONIN = os.path.join(FR_SOURCE_DIR, "Bonin2018.xlsx")
PATH_DESROCHERS = os.path.join(FR_SOURCE_DIR, "Desrochers2009.xls")

PATH_NORMS_DE = os.path.join(FIELD_DIR, "data.wordnorms_orig.de.csv")
PATH_VECNORMS_DE = os.path.join(FIELD_DIR, "data.wordnorms_vec.de.pkl.gz")
PATH_ALLNORMS_DE = os.path.join(FIELD_DIR, "data.allnorms.de.pkl.gz")
PATH_MODELS_DE = os.path.join(PATH_DATA, "models_de")
DE_SOURCE_DIR = os.path.join(SOURCE_DIR, "de_norms")
PATH_CONDE = os.path.join(DE_SOURCE_DIR, "Conde2026.xlsx")
PATH_SCHMIDTKE = os.path.join(DE_SOURCE_DIR, "Schmidtke2014.xlsx")
PATH_FREQS_DB = os.path.expanduser("~/lltk_data/data/metadb_freqs.duckdb")
PATH_SCORES_DB = os.path.join(SCORES_DIR, "scores.duckdb")
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
