#!/usr/bin/env python3
"""
Author:         David Meijer
Description:    Script for evaluating clustering results.
Usage:          For help use './01_parse_mibig.py -h'.
"""
import typing as ty
import argparse
import os

import numpy as np
import dill
from rdkit import Chem
import matplotlib.pyplot as plt
from pvclust import PvClust, seplot_with_heatmap


def cli() -> argparse.Namespace:
    """
    Command line interface for script.

    Returns
    -------
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to dilled clustering results file."
    )
    parser.add_argument(
        "-iraw",
        "--input_raw",
        type=str,
        required=True,
        help="Path to file containing raw data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output directory to write out results to."
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Results output identifier name."
    )
    return parser.parse_args()


def load_clustering_results(path: str) -> PvClust:
    """
    Load clustering results from file.

    Parameters
    ----------
    path (str): Path to clustering results file.

    Returns
    -------
    PvClust: Clustering results.
    """
    with open(path, "rb") as handle:
        return dill.load(handle)


def load_clustering_items(
    path: str
) -> ty.Tuple[ty.List[str], ty.List[Chem.Mol]]:
    """
    Load clustering items from file.

    Parameters
    ----------
    path (str): Path to file containing clustering items.

    Returns
    -------
    ty.Tuple[ty.List[str], ty.List[Chem.Mol]]: Tuple containing labels and
        mols.
    """
    with open(path, "rb") as handle:
        return zip(*dill.load(handle))


def main() -> None:
    """
    Driver code.
    """
    args = cli()
    pv = load_clustering_results(args.input)
    mol_labels, mols = load_clustering_items(args.input_raw)

    # Plot p-values vs. SEs.
    seplot_with_heatmap(
        pv,
        savefig=os.path.join(args.output, f"{args.name}_seplot.png")
    )
    # Get clusters of interest.
    cois = pv.result[(pv.result["AU"] >= 0.95) & (pv.result["SE.AU"] <= 0.1)]
    sizes = [len(pv.clusters[i]) for i in cois.index.values]
    print(f"{len(sizes)} clusters of interest.")
    plt.clf()
    plt.hist(sizes, color="grey", edgecolor="k", bins=100)
    plt.xlabel("Cluster sizes (AU p-value >= 0.95 & AU p-value SE <= 0.05)")
    plt.ylabel("Count")
    plt.savefig(
        os.path.join(args.output, f"{args.name}_cluster_size_1.png"),
        dpi=300
    )
    plt.clf()
    sizes = [s for s in sizes if s <= 20]
    labels, counts = np.unique(sizes, return_counts=True)
    labels = list(labels)
    counts = list(counts)
    for l in range(1, max(labels) + 1):
        if l not in labels:
            labels.append(l)
            counts.append(0)
    plt.xticks(labels)
    plt.bar(labels, counts, align="center", color="grey", edgecolor="k")
    plt.savefig(
        os.path.join(args.output, f"{args.name}_cluster_size_2.png"),
        dpi=300
    )
    plt.clf()

    # Plot tree with p-values.
    pv.plot(
        os.path.join(args.output, f"{args.name}_tree.png"),
        labels=mol_labels
    )

    # Extract a cluster of interest.
    clusters = [pv.clusters[i] for i in cois.index.values]
    print(clusters)


if __name__ == "__main__":
    main()
