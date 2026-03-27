"""
abstraction — Measuring abstract and concrete language in literary history.
"""

from .config import PATH_CORPORA, PATH_DATA
from .corpus import load_corpus, Corpus, pmap, pmap_iter
from .tokenize import tokenize, tokenize_agnostic, get_stopwords
from .norms import (
    get_orignorms, get_vecnorms, get_allnorms,
    get_origfields, get_allfields,
    get_origcontrasts, get_allcontrasts,
    format_norms_as_long, classify_word,
    gen_orignorms, corr_norms,
)
from .models import gen_vecnorms
from .counting import count_absconc, count_absconc_psg, count_absconc_corpus
from .scoring import score_psg, score_freqs, score_words, get_all_passages, gen_bookpassages, score_corpus_freqs, score_all_corpora
from .analysis import (
    load_scores, load_all_scored, adjust_scores,
    fit_arc, fit_arc_corpus, fit_arc_all_corpora,
    fit_arc_by_genre, fit_arc_all_by_genre, harmonize_genre, summarize_arc,
)
from .passages import (
    render_passage_html, display_passage, save_passage_html, save_passage_image,
    render_comparison_html, display_comparison, save_comparison_image,
)
from .words import (
    aggregate_freqs_by_decade, load_aggregate_freqs,
    correlate_words_with_trend, word_contributions,
    summarize_correlations, summarize_contributions,
)
from .utils import read_df, save_df, get_avgs_df
