import numpy as np
import pandas as pd
import logging
import sys

precision = 3  # for rounding, in python, numpy, pandas

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",  # "%(levelname)s:%(message)s"
)  # filename="log.log"
# logger.removeHandler(sys.stderr)


def score_match(f1, f2):
    """This function defines how to score the match between two frequencies."""
    # Currently implemented as the inverse of the relative difference
    # --> small difference = high score
    # TODO warning, division by zero possible
    # TODO should somehow score lower harmonics (simpler ratios) higher
    #      e.g. 5/7 is better than 33/35 or 3/999
    return abs(1 / ((f2 - f1) / f1))


def print_with_factor_and_fund_headers(mtx, f1, f2, harmonic_list):
    df = pd.DataFrame(mtx, columns=[f1, harmonic_list], index=[f2, harmonic_list])
    logger.info(f"Score for the common fundamental:\n{df}\n")


def print_with_freq_headers(mtx, freq_list):
    df = pd.DataFrame(
        np.where(mtx == 0, "-", np.round(mtx, precision)),
        columns=freq_list,
        index=freq_list,
    )
    print(f"{df}\n")


def merge_relative(numbers, threshold):
    if not numbers:
        return []

    numbers.sort()  # Ensure numbers are in ascending order
    merged = []
    groups = []  # Stores the groups of merged numbers
    group = [numbers[0]]  # Start with the first number

    for num in numbers[1:]:
        if abs(num - group[-1]) / group[-1] <= threshold:  # Relative difference check
            group.append(num)
        else:
            merged.append(sum(group) / len(group))  # Merge (average)
            groups.append(group[:])  # Store the original group
            group = [num]  # Start new group

    merged.append(sum(group) / len(group))  # Merge last group
    groups.append(group)  # Store the last group

    return list(zip(groups, merged))


def find_fundamentals(
    freqs, max_harmonic, distance_threshold, similarity_threshold, even=False
):

    freqs = sorted(freqs)
    num_freqs = len(freqs)
    if len(freqs) < 2:
        raise Exception("Must provide at least two frequencies")

    harmonic_list = list(range(1, max_harmonic + 1, 2))
    num_harmonics = len(harmonic_list)
    # TODO implement even harmonics too
    if even:
        raise Exception("Even harmonics not yet implemented")

    # setup some output options
    np.set_printoptions(linewidth=200, suppress=True, precision=precision)
    max_needed_cells = max(num_harmonics + 2, num_freqs + 1)
    pd.set_option("display.max_rows", max_needed_cells)
    pd.set_option("display.max_columns", max_needed_cells)
    pd.set_option("display.precision", precision)

    # initialize result matrices
    best_harmonic_avg_mtx = np.zeros((num_freqs, num_freqs))
    best_harmonic_score_mtx = np.zeros((num_freqs, num_freqs))

    for i1, freq1 in enumerate(freqs):
        # generate candidates for fundamental frequency
        funds1 = [freq1 / n for n in harmonic_list]
        offset_for_i2 = i1 + 1  # only fill upper right triangle of result matrix
        for i2_fromzero, freq2 in enumerate(freqs[offset_for_i2:]):
            i2 = i2_fromzero + offset_for_i2  # match iterator index to matrix index
            funds2 = [freq2 / n for n in harmonic_list]

            logger.debug(f"Comparing {freq1} and {freq2}")
            # score all combination of the candidates for common fundamental
            score_mtx = score_match(*np.meshgrid(funds1, funds2, indexing="xy"))
            #                                             for column first ^
            print_with_factor_and_fund_headers(score_mtx, funds1, funds2, harmonic_list)

            # get maximum score and corresponding position in matrix
            # TODO: it's super naive, figure out a better way to prefer smaller divisors
            max_score = np.max(score_mtx)
            max_idx = np.argmax(score_mtx)
            max_idx2d = np.unravel_index(max_idx, score_mtx.shape, order="F")
            #                                            for column first ^

            # convert from indices (0,1,2...) to corresponding harmonic multiples (1,3,5,...)
            multiple1 = harmonic_list[max_idx2d[0]]
            multiple2 = harmonic_list[max_idx2d[1]]
            fund1, fund2 = freq1 / multiple1, freq2 / multiple2

            best_harmonic_avg_mtx[i1, i2] = (fund1 + fund2) / 2
            best_harmonic_score_mtx[i1, i2] = max_score

            logger.info(
                f"Most likely common fundamental for {freq1} and {freq2}: approx {round((fund1+fund2)/2, precision)} (averaged)"
            )
            logger.info(f"{freq1} = {multiple1} x {round(fund1,precision)}")
            logger.info(f"{freq2} = {multiple2} x {round(fund2,precision)}")
            logger.info(f"Score: {round(max_score,3)}\n")

    print("Final results:")

    print("Most likely common fundamental:")
    print_with_freq_headers(best_harmonic_avg_mtx, freqs)

    print("Confidence score for most likely fundamental:")
    print_with_freq_headers(best_harmonic_score_mtx, freqs)

    best_harmonic_avg_filtered_mtx = np.where(
        best_harmonic_score_mtx < distance_threshold, 0, best_harmonic_avg_mtx
    )
    print(
        f"Most likely common fundamental:\n"
        f"(if score is above threshold of {distance_threshold})"
    )
    print_with_freq_headers(best_harmonic_avg_filtered_mtx, freqs)

    funds_list = [f for f in best_harmonic_avg_filtered_mtx.ravel() if f > 0]

    print("List of candidates for fundamentals:")
    for f in funds_list:
        print(f"- {round(f,precision)}")

    merged = merge_relative(funds_list, similarity_threshold)
    reduced = []
    for group, merged_value in merged:
        logger.debug(
            f"Group: {np.array(group)} → Merged into: {round(merged_value,precision)}"
        )
        reduced.append(merged_value)
    print()

    print(
        f"Reduced list of candidates (similarity threshold: {similarity_threshold*100}%):"
    )
    for f in reduced:
        print(f"- {round(f,precision)}")
