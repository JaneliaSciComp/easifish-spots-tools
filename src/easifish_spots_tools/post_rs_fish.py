import argparse
import numpy as np
import os

from .cli import floattuple
from .io_utils.read_utils import read_array_attrs
from zarr_tools.ngff.ngff_utils import get_spatial_dataset_voxel_spacing


def _define_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-i', '--input',
                             dest='input',
                             type=str,
                             required=True,
                             help = "spots input file using voxel coordinates")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             required=True,
                             help = "output file in real coordinates")

    args_parser.add_argument('--image-container',
                             dest='image_container',
                             type=str,
                             help = "image container")
    args_parser.add_argument('--image-subpath', '--image-dataset',
                             dest='image_dataset',
                             type=str,
                             help = "image subpath")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             metavar='SX,SY,SZ',
                             help = "Voxel spacing")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    args_parser.add_argument('--ignore-voxel-spacing',
                             dest='ignore_voxel_spacing',
                             action='store_true',
                             default=False,
                             help='Do not apply voxel spacing')

    return args_parser


def _post_process_rsfish_csv_results(args):

    # the coordinates in the spots file are as x,y,z
    # so we need to ensure the voxel spacing is also as sx,sy,sz
    if args.voxel_spacing:
        # leave the voxel spacing as (X,Y,Z)
        voxel_spacing = np.array(args.voxel_spacing)
    elif args.image_container:
        image_attrs = read_array_attrs(args.image_container, args.image_dataset)
        print(f'Image attributes: {image_attrs}')
        voxel_spacing = get_spatial_dataset_voxel_spacing(image_attrs, args.image_dataset)
        if voxel_spacing is not None:
            # voxel spacing is returned as SZ,SY,SX so revert it
            voxel_spacing = voxel_spacing[::-1]
    else:
        voxel_spacing = None

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            voxel_spacing = [c / args.expansion_factor for c in voxel_spacing]
    else:
        voxel_spacing = (1,) * 3

    print(f'Image voxel spacing (sx.sy,sz) for dataset {args.image_dataset}: {voxel_spacing}')

    with open(args.input, 'r') as f:
        first_line = f.readline().strip()

    # check if the first line is a header (non-numeric) or data
    try:
        np.array(first_line.split(','), dtype=float)
        header = None
        skiprows = 0
    except ValueError:
        header = first_line
        skiprows = 1

    rsfish_spots = np.loadtxt(args.input, delimiter=',', skiprows=skiprows)
    if len(rsfish_spots) == 0:
        print(f'No spots found in {args.input}')
        return

    if not args.ignore_voxel_spacing:
        rsfish_spots[:, :3] = rsfish_spots[:, :3] * voxel_spacing

    # extract unique channels
    rsfish_channels = np.unique(rsfish_spots[:, 4]).astype(int)
    if len(rsfish_channels) == 1:
        # the RS-FISH result only has one channel
        _save_results_per_channel(rsfish_spots, args.output, header)
    else:
        name, ext = os.path.splitext(args.output)
        print(f'{args.output} -> {name}, {ext}')
        for c in rsfish_channels:
            channel_result_file = f'{name}-{c-1}{ext}'
            _save_results_per_channel(rsfish_spots[rsfish_spots[:,4] == c], channel_result_file, header)


def _save_results_per_channel(rsfish_spots, res_file, header):
    # Remove unnecessary columns (t,c) at indexes 3 and 4
    rsfish_spots = np.delete(rsfish_spots, np.s_[3:5], axis=1)

    print(f'Saving {rsfish_spots.shape} points in micron space to {res_file}')
    fmt = ['%.4f'] * rsfish_spots.shape[1]

    if header:
        # Remove t,c columns from header as well
        header_cols = header.split(',')
        header_cols = [c for i, c in enumerate(header_cols) if i not in (3, 4)]
        header = ','.join(header_cols)
        np.savetxt(res_file, rsfish_spots, delimiter=',', fmt=fmt, header=header, comments='')
    else:
        np.savetxt(res_file, rsfish_spots, delimiter=',', fmt=fmt)


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # run post processing
    _post_process_rsfish_csv_results(args)


if __name__ == '__main__':
    _main()
