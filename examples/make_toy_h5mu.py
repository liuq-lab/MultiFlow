"""Create a tiny paired H5MU file for the MultiFlow quick start."""

from __future__ import annotations

import argparse

from multiflow.h5mu import write_toy_h5mu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="toy_multiflow.h5mu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    destination = write_toy_h5mu(args.output, seed=args.seed)
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
