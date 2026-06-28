# preprocessing.py

import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# INISIALISASI (hanya sekali saat import)
# =====================================================

_stop_factory    = StopWordRemoverFactory()
_stopword_list   = set(_stop_factory.get_stop_words())
_stemmer_factory = StemmerFactory()
_stemmer         = _stemmer_factory.create_stemmer()

# =====================================================
# FUNGSI PREPROCESSING
# =====================================================

def case_folding(text: str) -> str:
    return text.lower().strip()

def cleaning(text: str) -> str:
    text = re.sub(r'[^a-z\s]', ' ', text)   # hapus angka, tanda baca, karakter aneh
    text = re.sub(r'\s+', ' ', text)          # hapus spasi berlebih
    return text.strip()

def remove_stopwords(text: str) -> str:
    words = text.split()
    words = [w for w in words if w not in _stopword_list]
    return ' '.join(words)

def stemming(text: str) -> str:
    return _stemmer.stem(text)

def preprocess(judul: str, ringkasan: str) -> str:
    """
    Pipeline lengkap preprocessing untuk input Streamlit.
    Menggabungkan judul + ringkasan menjadi 1 Final_Text.
    """
    text = f"{judul} {ringkasan}"
    text = case_folding(text)
    text = cleaning(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text