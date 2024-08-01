from dataloader.synthetic_dataloader import SynthDataloader
import argparse
import matplotlib.pyplot as plt

def main(args):
    dataloader = SynthDataloader(
        evaluation_indices=args.eval_img_inds,
        data_path=args.data_path,
        camera_focal_lenth=args.focal_length,
        map_pointcloud_cache_path=args.map_pcd_cache_path
    )

    rgb, _, _ = dataloader.get_image_data(0)
    
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to synthetic data",
        default="./data/our-synthetic/360_basic_test"
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
        "--focal-length",
        type=float,
        help="Focal length of camera",
        default=300
    )
    parser.add_argument(
        "--map-pcd-cache-path",
        type=str,
        help="Location where the map's pointcloud is cached for future use",
        default="./cache/360_zip_cache_map.pcd"
    )
    args = parser.parse_args()

    main(args)