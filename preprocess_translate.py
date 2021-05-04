INDIC_NLP_LIB_HOME = "indic_nlp_library"
INDIC_NLP_RESOURCES = "indic_nlp_resources"
import sys

sys.path.append(r"{}".format(INDIC_NLP_LIB_HOME))
from indicnlp import common

common.set_resources_path(INDIC_NLP_RESOURCES)
from indicnlp import loader

loader.load()
from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer
from sacremoses import MosesDetokenizer
from collections import defaultdict

from tqdm import tqdm
from joblib import Parallel, delayed

from indicnlp.tokenize import indic_tokenize
from indicnlp.tokenize import indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate

# for vietnamese tokenization
from vncorenlp import VnCoreNLP
from clean_vi_text import fix_contents
import re

import tnkeeh as tn
from farasa.segmenter import FarasaSegmenter


en_tok = MosesTokenizer(lang="en")
en_normalizer = MosesPunctNormalizer()
# TODO: change hardcoding of jar file to a arg from cli
rdrsegmenter = VnCoreNLP(
    "vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size="-Xmx500m"
)
ar_segmenter = FarasaSegmenter()


def clean_ar_text(
    text,
    segment=False,
    remove_special_chars=False,
    remove_english=False,
    normalize=False,
    remove_diacritics=False,
    excluded_chars=[],
    remove_tatweel=False,
    remove_html_elements=False,
    remove_links=False,
    remove_twitter_meta=False,
    remove_long_words=False,
    remove_repeated_chars=False,
    by_chunk=False,
    chunk_size=100000,
    normalize_dots=False,
):

    if remove_repeated_chars:
        text = tn.tnkeeh._remove_repeated_chars(text)
    if remove_html_elements:
        text = tn.tnkeeh._remove_html_elements(text)
    if segment:
        text = tn.tnkeeh._farasa_segment(text, ar_segmenter)
    if remove_english:
        text = tn.tnkeeh._remove_english_chars(text)
    if normalize:
        text = tn.tnkeeh._normalize_data(text)
    if remove_diacritics:
        text = tn.tnkeeh._remove_diacritics(text)
    if remove_special_chars:
        text = tn.tnkeeh._remove_special_chars(text, excluded_chars)
    if remove_tatweel:
        text = re.sub("ـ", "", text)
    if remove_links:
        text = tn.tnkeeh._remove_links(text)
    if remove_twitter_meta:
        text = tn.tnkeeh._remove_twitter_meta(text)
    if remove_long_words:
        text = tn.tnkeeh._remove_long_words(text)

    text = tn.tnkeeh._add_spaces_to_all_special_chars(text)
    text = tn.tnkeeh._remove_extra_spaces(text)
    return text


def preprocess_line(line, normalizer, lang, transliterate=False):
    if lang == "en":
        # this is using cleaner for vi text and imp for en-vi dataset
        # TODO: how to not include this for other language cleaning
        # line = fix_contents(line)
        return " ".join(
            en_tok.tokenize(en_normalizer.normalize(line.strip()), escape=False)
        )
    elif lang == "vi":
        line = fix_contents(line)
        sentences = rdrsegmenter.tokenize(line)
        tokenized_sentence = ""
        # if len(sentences) > 1:
        #     print(
        #         f"WARNING: Vietnamese line is getting segmented to multiple sentences. Recheck: {line}"
        #     )
        for sentence in sentences:
            tokenized_sentence += " ".join(sentence)
            if sentences[-1] != sentence:
                tokenized_sentence += " "
        return tokenized_sentence

    elif transliterate:
        # line = indic_detokenize.trivial_detokenize(line.strip(), lang)
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
            " ".join(
                indic_tokenize.trivial_tokenize(
                    normalizer.normalize(line.strip()), lang
                )
            ),
            lang,
            "hi",
        ).replace(" ् ", "्")
    else:
        # we only need to transliterate for joint training
        return " ".join(
            indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), lang)
        )


def preprocess(infname, outfname, lang, transliterate=False):
    """
    Normalize, tokenize and script convert(for Indic)
    return number of sentences input file

    """

    n = 0
    num_lines = sum(1 for line in open(infname, "r"))
    if lang == "en" or lang == "vi":
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, None, lang)
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1
    elif lang == "ar":
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(clean_ar_text)(
                    text=line,
                    remove_diacritics=True,
                    segment=True,
                    normalize=True,
                )
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1
    else:
        normfactory = indic_normalize.IndicNormalizerFactory()
        normalizer = normfactory.get_normalizer(lang)
        # reading
        with open(infname, "r", encoding="utf-8") as infile, open(
            outfname, "w", encoding="utf-8"
        ) as outfile:

            out_lines = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(preprocess_line)(line, normalizer, lang, transliterate)
                for line in tqdm(infile, total=num_lines)
            )

            for line in out_lines:
                outfile.write(line + "\n")
                n += 1
    return n


def old_preprocess(infname, outfname, lang):
    """
    Preparing each corpus file:
      - Normalization
      - Tokenization
      - Script coversion to Devanagari for Indic scripts
    """
    n = 0
    num_lines = sum(1 for line in open(infname, "r"))
    # reading
    with open(infname, "r", encoding="utf-8") as infile, open(
        outfname, "w", encoding="utf-8"
    ) as outfile:

        if lang == "en":
            en_tok = MosesTokenizer(lang="en")
            en_normalizer = MosesPunctNormalizer()
            for line in tqdm(infile, total=num_lines):
                outline = " ".join(
                    en_tok.tokenize(en_normalizer.normalize(line.strip()), escape=False)
                )
                outfile.write(outline + "\n")
                n += 1

        else:
            normfactory = indic_normalize.IndicNormalizerFactory()
            normalizer = normfactory.get_normalizer(lang)
            for line in tqdm(infile, total=num_lines):
                outline = (
                    unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                        " ".join(
                            indic_tokenize.trivial_tokenize(
                                normalizer.normalize(line.strip()), lang
                            )
                        ),
                        lang,
                        "hi",
                    ).replace(" ् ", "्")
                )

                outfile.write(outline + "\n")
                n += 1
    return n


if __name__ == "__main__":

    # INDIC_NLP_LIB_HOME = "indic_nlp_library"
    # INDIC_NLP_RESOURCES = "indic_nlp_resources"
    # sys.path.append(r'{}'.format(INDIC_NLP_LIB_HOME))
    # common.set_resources_path(INDIC_NLP_RESOURCES)

    # data_dir = '../joint_training/v1'
    # new_dir = data_dir + '.norm'
    # for path, subdirs, files in os.walk(data_dir):
    #     for name in files:
    #         infile = os.path.join(path, name)
    #         lang = infile.split('.')[-1]
    #         outfile = os.path.join(path.replace(data_dir, new_dir), name)
    #         preprocess(infile, outfile, lang)
    # loader.load()

    infname = sys.argv[1]
    outfname = sys.argv[2]
    lang = sys.argv[3]

    if len(sys.argv) == 4:
        transliterate = False
    elif len(sys.argv) == 5:
        transliterate = sys.argv[4]
        if transliterate.lower() == "true":
            transliterate = True
        else:
            transliterate = False
    else:
        print(f"Invalid arguments: {sys.argv}")
        exit()
    if lang == "ar":
        tn.clean_data(
            file_path=infname,
            save_path=outfname,
            remove_diacritics=True,
            segment=True,
            normalize=True,
        )
    else:
        print(preprocess(infname, outfname, lang, transliterate))
