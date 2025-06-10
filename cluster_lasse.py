import os

import pickle
import numpy as np
from tqdm import tqdm
import typer
from pathlib import Path

def main(latent: Path, composition: Path, cuda: bool, mrad="0.5", drad="0.8"):

    os.environ['MEDOID_RADIUS'] = mrad
    os.environ['DEFAULT_RADIUS'] = drad

    # Read in relevant data
    ar = np.load(latent)
    latent = ar["arr_0"]
    meta = np.load(composition, allow_pickle=True)
    sequence_lens = meta["lengths"]
    sequence_names = meta["identifiers"]

    cluster( base_clusters_name = f"different_clusters/test_clusters_mrad_{mrad}_defrad_{drad}.tsv", cuda=cuda, latent=latent, sequence_names=sequence_names, sequence_lens=sequence_lens)

def cluster(  base_clusters_name = "clusters.tsv", cuda = False, latent=None, sequence_names=None, sequence_lens=None):


    import vamb
    from vamb_file import write_clusters_and_bins, ClusterOptions, itertools, cast

    cluster_options = ClusterOptions(window_size = 300, min_successes = 15, max_clusters=None)
    cluster_generator = vamb.cluster.ClusterGenerator(
        latent,
        sequence_lens,
        windowsize=cluster_options.window_size,
        minsuccesses=cluster_options.min_successes,
        destroy=True,
        normalized=False,
        cuda=cuda,
        rng_seed=42,
    )
    # This also works correctly when max_clusters is None
    clusters = itertools.islice(cluster_generator, cluster_options.max_clusters)
    cluster_dict: dict[str, set[str]] = dict()

    # Write the cluster metadata to file
    for i, cluster in tqdm(enumerate(clusters)):
        cluster_dict[str(i + 1)] = {
            sequence_names[cast(int, i)] for i in cluster.members
        }

    fasta_output = None
    bin_prefix = None
    binsplitter = vamb.vambtools.BinSplitter("C")
    cluster_dic = cluster_dict,

    write_clusters_and_bins(
        fasta_output,
        bin_prefix,
        binsplitter,
        base_clusters_name,
        cluster_dict,
        sequence_names,
        sequence_lens
    )


if __name__ == "__main__":
    typer.run(main)
