#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import click
import sys

import fundamentalfinder as ff


@click.command()
@click.option(
    "-s",
    "--separator",
    type=str,
    default="\n",
    # show_default=True,
    help="Separator of frequency list [default: '\\n', fallback: ',']",
)
@click.option(
    "-h",
    "--max-harmonic",
    type=int,
    default=15,
    show_default=True,
    help="Maximum harmonic multiple to analyze",
)
@click.option(
    "-d",
    "--distance-threshold",
    type=int,
    default=1000,
    show_default=True,
    help="Threshold of matching score",
)
@click.option(
    "-s",
    "--similarity-threshold",
    type=float,
    default=0.01,
    show_default=True,
    help="Threshold of similarity when grouping fundamental candidates",
)
@click.option(
    "-e",
    "--even",
    is_flag=True,
    default=False,
    show_default=True,
    help="Also include even frequencies",
)
def fundamentalfinder(
    separator, max_harmonic, distance_threshold, similarity_threshold, even
):
    freqs = sys.stdin.read().split(separator)
    freqs = [f for f in freqs if not f == ""]
    # hack / for comfort: use comma as second fallback default separator
    # if input is still one line after splitting, try with comma
    if len(freqs) == 1:
        freqs = freqs[0].split(",")
    freqs = [f.strip() for f in freqs]
    freqs = [float(f) for f in freqs if not f == ""]
    freqs = sorted(freqs)  # redundant, but best to keep in both plpaces

    print(
        f"Analyzing the following {len(freqs)} frequencies up to the {max_harmonic}th harmonic:"
    )
    for f in freqs:
        print(f"- {f}")
    # print(f"Score threshold: {distance_threshold}")
    # print(f"Similarity threshold: {similarity_threshold*100}%")
    print()

    ff.find_fundamentals(
        freqs, max_harmonic, distance_threshold, similarity_threshold, even
    )


if __name__ == "__main__":
    fundamentalfinder()
