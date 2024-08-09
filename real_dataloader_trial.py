from dataloader.real_dataloader import RealDataloader
import argparse
import matplotlib.pyplot as plt

def main(args):
    dataloader = RealDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        focal_length_x=args.focal_length_x,
        focal_length_y=args.focal_length_y,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    rgb, _, _ = dataloader.get_image_data(1)
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to synthetic data",
        default="./data/new_lab"
    )
    parser.add_argument(
        "-e",
        "--eval-img-inds",
        type=int,
        nargs='+',
        help="Indices to be evaluated",
        default=[4]
    )
    parser.add_argument(
        "--focal-length-x",
        type=float,
        help="x-Focal length of camera",
        default=385.28887939453125
    )
    parser.add_argument(
        "--focal-length-y",
        type=float,
        help="y-Focal length of camera",
        default=384.3631591796875
    )
    parser.add_argument(
        "--map-pcd-cache-path",
        type=str,
        help="Location where the map's pointcloud is cached for future use",
        default="./.cache/gf_lab4_cache.pcd"
    )
    args = parser.parse_args()

    main(args)