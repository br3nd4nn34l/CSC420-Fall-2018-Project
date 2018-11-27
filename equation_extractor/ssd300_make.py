import os
import sys

# So this can be run as a script
sys.path.append(os.path.dirname(sys.path[0]))

import argparse

def make_network(size, destination):
    from equation_extractor.ssd_helpers import make_ssd300_model

    print(f"Creating the architecture for an SSD300 model to "
          f"detect equation bounding boxes for:"
          f"\n\t{size}x{size} Input Images"
          f"\n\t1             Class (Equations)")
    print("Saving architecture to {}".format(destination))

    # Build model for equation finding task
    model = make_ssd300_model(size)

    # Report model architecture and save architecture
    model.summary()
    with open(destination, "w") as file:
        file.write(model.to_json())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=f"Creates an SSD300 model architecture optimized for detecting"
                    f"equation-bounding-boxes for k x k square images."
    )

    parser.add_argument(
        "size",
        type=int,
        help="Side length (in pixels) of the square input images to the network."
    )

    parser.add_argument(
        "destination",
        type=str,
        help="Where to save the model's JSON architecture."
    )

    args = parser.parse_args()
    make_network(
        size=args.size,
        destination=args.destination,
    )
