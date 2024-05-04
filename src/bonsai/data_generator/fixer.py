# python3 fixer.py raw/MAY/ fixed/ MAY 2>&1 | tee result

"""
This script is used to fix the output produced by ChatGPT. It basically uses
a sentence-comparison algorithm to find the most similar sentence to a
particular sentence and groups them accordingly.
"""

from pathlib import Path
from os import walk
from json import load, dumps
from re import compile as re_compile, IGNORECASE
from sys import argv
from logging import (
    getLogger,
    Formatter,
    StreamHandler,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL
)
from ipdb import set_trace
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy import dot
from time import time

DEBUG
INFO
WARNING
ERROR
CRITICAL

set_trace

download
# download('punkt')
# download('stopwords')
stop_words = set(stopwords.words('english'))
vectorizer = TfidfVectorizer()


class ColorFormatter(Formatter):

    colors = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m'  # Red
    }

    def format(self, record: str) -> None:
        return (
            f"{self.colors[record.levelname]}"
            f"{super().format(record)}\033[0m"
        )


equivalent_re = re_compile(r".*equivalent: ", IGNORECASE)
may_in_double_quotes = re_compile(r'"may"', IGNORECASE)
more_than_one_may_or_must_re = re_compile(
    r"\b(may|must)\b(?=.*?\b\1\b)", IGNORECASE
)
may_or_must_re = re_compile(r".*(\bmay\b|\bmust\b).*", IGNORECASE)
strip_re = re_compile(
    r"\A(?P<lead>[^a-zA-Z]*)?(?P<line>.*?)(?P<trail>\s*)?\Z"
)
quote_re = re_compile(r"\\u201[89]")
non_alphanumeric_lead_trail_re = re_compile(r"^(?:\W+)?(.*?)(?:\W+)?$")

logger = getLogger(__file__)
# logger.setLevel(DEBUG)
logger.setLevel(INFO)
# logger.setLevel(WARNING)
# logger.setLevel(ERROR)
# logger.setLevel(CRITICAL)
console_handler = StreamHandler()
console_handler.setFormatter(ColorFormatter(fmt="%(levelname)s - %(message)s"))
logger.addHandler(console_handler)


def seconds_to_hms(seconds: int) -> str:

    return (
        f"{int(seconds // 3600)}:"
        f"{int((seconds % 3600) // 60)}:"
        f"{round(seconds % 60)}"
    )


def preprocess(sentence: str) -> str:
    return " ".join(
        [
            word for word in word_tokenize(sentence.lower())
            if word.isalnum() and word not in stop_words
        ]
    )


def calculate_similarity(first_line: str, second_line: str) -> float:

    vectors = vectorizer.fit_transform(
        [
            preprocess(first_line),
            preprocess(second_line)
        ]
    )

    return dot(vectors, vectors.T).toarray()[0, 1]


if __name__ == "__main__":

    total_time_start = time()

    total_line_counter = 0
    accepted_line_pair_counter = 0
    rejected_line_counter = 0

    result = set()

    file_times_per_line = []
    raw_directory_path = Path(argv[1])
    fixed_directory_path = Path(argv[2])
    verb = argv[3]

    for file_name in sorted(next(walk(raw_directory_path))[2]):

        file_time_start = time()

        logger.info(f"{file_name} processing starts")

        with open(raw_directory_path.joinpath(file_name), "r") as file_file:
            try:
                lines = load(file_file)
                len_lines = len(lines)
                total_line_counter += len_lines
            except Exception:
                logger.error(f"{file_name} unable to load")
                continue

        if not lines:
            logger.error(f"{file_name} is empty")
            continue

        if (
            len(lines) == 2 or len(lines) == 3
        ) and (
            lines[0].count(".") > 1
        ) and (
            lines[1].count(".") > 1
        ):

            new_lines = []

            new_lines.extend(lines[0].split("."))
            new_lines.extend(lines[1].split("."))

            if len(lines) == 3:

                new_lines.append(lines[2])

            lines = new_lines

        if may_or_must_re.match(lines[-1]) is not None:
            lines = lines[:-1]

        may_or_must_lines = list()
        no_may_or_must_lines = set()

        paired_lines = list()

        for line in lines:

            if (
                may_in_double_quotes.search(line) or
                len(set(list(line))) == 1 or
                not line
            ):
                continue

            line = quote_re.sub("'", line)
            line = (
                non_alphanumeric_lead_trail_re.match(line).group(1)
            )
            line = equivalent_re.sub("", line)
            line = strip_re.match(line).group("line")

            if may_or_must_re.match(line) is None:
                no_may_or_must_lines.add(line)

            else:
                may_or_must_lines.append(line)

        for may_or_must_line in [
            may_or_must_line for may_or_must_line in may_or_must_lines
            if (
                more_than_one_may_or_must_re.search(may_or_must_line)
                is None
            )
        ]:

            if not no_may_or_must_lines:
                logger.warning(
                    f"{file_name} consumed all no_may_or_must_lines"
                )
                break

            most_similar_no_may_or_must_line = None
            greatest_similarity = 0

            for no_may_or_must_line in no_may_or_must_lines:

                similarity = calculate_similarity(
                    may_or_must_line, no_may_or_must_line
                )

                if similarity > greatest_similarity:
                    greatest_similarity = similarity
                    most_similar_line = no_may_or_must_line

            if greatest_similarity > 0.5:

                no_may_or_must_lines.remove(most_similar_line)

                pair = (most_similar_line, may_or_must_line)

                paired_lines.append(pair)
                result.add(pair)

                accepted_line_pair_counter += 1

                logger.debug(
                    f"{file_name} Added: {pair[0]} | {pair[1]}"
                )
            else:
                rejected_line_counter += 1
                logger.warning(
                    f"{file_name} Rejected: {may_or_must_line} "
                )

        with open(
            fixed_directory_path.joinpath(f"{file_name}"),
            "w"
        ) as fixed_file_file:
            fixed_file_file.write(
                dumps(paired_lines, indent=4, ensure_ascii=False)
            )
            fixed_file_file.write("\n")

        file_time_end = time()

        logger.info(f"{file_name} processing ends")

        file_time = file_time_end - file_time_start

        file_time_per_line = file_time / len_lines
        file_times_per_line.append(file_time_per_line)

        logger.critical(f"{file_name} took {round(file_time, 4)}s")
        logger.critical(
            f"{file_name} took {round(file_time_per_line, 4)}s/l"
        )

    if result:

        result_file_path = fixed_directory_path.joinpath(f"{verb}.json")

        with open(result_file_path, "w") as result_file_file:
            result_file_file.write(
                dumps(list(result), indent=4, ensure_ascii=False)
            )
            result_file_file.write("\n")

    total_time_end = time()

    print()

    logger.critical(f"Half total lines: {total_line_counter / 2}")
    logger.critical(
        f"Rejected pairs of lines: {rejected_line_counter}"
    )
    logger.critical(
        f"Accepted pairs of lines: {accepted_line_pair_counter}"
    )

    total_line_pairs = (
        accepted_line_pair_counter + rejected_line_counter
    )

    logger.critical(f"Total pairs of lines: {total_line_pairs}")
    logger.critical(
        f"Rejection percentage: "
        f"{round(100 * rejected_line_counter / total_line_pairs, 2)}%"
    )
    logger.critical(
        f"Total time: "
        f"{seconds_to_hms(total_time_end - total_time_start)}"
    )
    logger.critical(
        f"Average time per line: "
        f"{round(sum(file_times_per_line) / len(file_times_per_line), 4)}s/l"
    )
