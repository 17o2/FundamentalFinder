import numpy as np
import pandas as pd

# inputs
# freqs = [125, 75.001]  # basic obvious example with one clear common fundamental
# freqs = [147.51, 206.47, 265.6]  # real life example
freqs = [147.51, 206.47, 265.6, 125, 75.001]  # combined, to test threshold
# freqs = [300, 307]  # no obvious shared fundamental

max_harmonic = 11  # should be uneven integer
score_threshold = 1000  # TODO: totally arbitrary right now

# clean up inputs
freqs = sorted(freqs)
harmonic_list = list(range(1, max_harmonic + 1, 2))

precision = 3  # for rounding, in python, numpy, pandas
np.set_printoptions(linewidth=200, suppress=True, precision=precision)
pd.set_option("display.max_rows", len(harmonic_list) + 2)
pd.set_option("display.max_columns", len(harmonic_list) + 2)
pd.set_option("display.precision", precision)


def score_match(f1, f2):
    """This function defines how to score the match between two frequencies."""
    # Currently implemented as the inverse of the relative difference
    # --> small difference = high score
    # TODO warning, division by zero possible
    # TODO should somehow score lower harmonics (simpler ratios) higher
    #      e.g. 5/7 is better than 33/35 or 3/999
    return abs(1 / ((f2 - f1) / f1))


def print_with_factor_and_fund_headers(mtx, f1, f2):
    df = pd.DataFrame(mtx, columns=[f1, harmonic_list], index=[f2, harmonic_list])
    print("Score for the common fundamental:")
    print(df)
    print()


def print_with_freq_headers(mtx):
    df = pd.DataFrame(mtx, columns=freqs, index=freqs)
    print(df)
    print()


if len(freqs) < 2:
    raise Exception("Must provide at least two frequencies")

# initialize result matrices
best_harmonic_avg_mtx = np.zeros((len(freqs), len(freqs)))
best_harmonic_score_mtx = np.zeros((len(freqs), len(freqs)))

print()
print(
    f"Will compare the following {len(freqs)} frequencies up to the {max_harmonic}th harmonic:"
)
print(freqs)
print()

for i1, freq1 in enumerate(freqs):
    # generate candidates for fundamental frequency
    funds1 = [freq1 / n for n in harmonic_list]
    offset_for_i2 = i1 + 1  # only fill upper right triangle of result matrix
    for i2_fromzero, freq2 in enumerate(freqs[offset_for_i2:]):
        i2 = i2_fromzero + offset_for_i2  # match iterator index to matrix index
        funds2 = [freq2 / n for n in harmonic_list]

        print(f"Comparing {freq1} and {freq2}")
        # score all combination of the candidates for common fundamental
        score_mtx = score_match(*np.meshgrid(funds1, funds2, indexing="xy"))
        #                                             for column first ^
        print_with_factor_and_fund_headers(score_mtx, funds1, funds2)

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

        print(
            f"Most likely common fundamental for {freq1} and {freq2}: approx {round((fund1+fund2)/2, precision)} (averaged)"
        )
        print(f"{freq1} = {multiple1} x {round(fund1,precision)}")
        print(f"{freq2} = {multiple2} x {round(fund2,precision)}")
        print(f"Score: {round(max_score,3)}")
        print()

print("--------------")
print("Final results:")
print()

print("Most likely common fundamental:")
print_with_freq_headers(best_harmonic_avg_mtx)

print("Confidence score for most likely fundamental:")
print_with_freq_headers(best_harmonic_score_mtx)

best_harmonic_avg_filtered_mtx = np.where(
    best_harmonic_score_mtx < score_threshold, 0, best_harmonic_avg_mtx
)
print(
    f"Most likely common fundamental:\n"
    f"(if score is above threshold of {score_threshold})"
)
print_with_freq_headers(best_harmonic_avg_filtered_mtx)

print()
