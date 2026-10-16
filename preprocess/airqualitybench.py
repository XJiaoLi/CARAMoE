from __future__ import annotations

import argparse

from data_provider.airqualitybench import compute_capability_profile
from lib.graph import load_or_build_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--metadata", default="./data/selected_nodes_metadata.csv")
    parser.add_argument("--capability-output", default="./data/capability_train_2021_2023.npz")
    parser.add_argument("--graph-output", default="./data/caramoe_graph_k10.npz")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    capability = compute_capability_profile(
        args.data_dir,
        years=(2021, 2022, 2023),
        output_path=args.capability_output,
    )
    neighbors, distances, metadata = load_or_build_graph(
        args.metadata,
        args.graph_output,
        k=args.k,
    )
    print(
        f"prepared capability={capability['coverage'].shape}, "
        f"graph={neighbors.shape}, stations={len(metadata)}"
    )


if __name__ == "__main__":
    main()
