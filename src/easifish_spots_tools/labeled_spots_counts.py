import argparse
import logging
import numpy as np
import os
import pandas as pd
import traceback

from glob import glob
from zarr_tools.io.zarr_io import read_zarr_block
from zarr_tools.ngff.ngff_utils import get_spatial_voxel_spacing


from .cli import floattuple
from .io_utils.read_utils import open_array, read_array_attrs
from .utils.configure_logging import configure_logging


logger:logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--labels-container',
                             dest='labels_container',
                             type=str,
                             required=False,
                             help = "path to the labels container")
    args_parser.add_argument('--labels-subpath', '--labels-dataset',
                             dest='labels_dataset',
                             type=str,
                             required=False,
                             help = "path to the labels container")

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
                             help = "voxel spacing")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    args_parser.add_argument('--spots-pattern',
                             dest='spots_pattern',
                             type=str,
                             required=True,
                             help = "Glob pattern for spots files")

    args_parser.add_argument('--timeindex',
                             dest='spots_timeindex',
                             type=int,
                             help = "Time index from the labels OME ZARR used for spots counting")

    args_parser.add_argument('--channel',
                             dest='spots_channel',
                             type=int,
                             help = "Channel index from the labels OME ZARR used for spots counting")

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             required=True,
                             help = "output file")

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='python log file configuration')
    args_parser.add_argument('--verbose',
                             action='store_true',
                             help="save debug level information to the log")

    return args_parser


def _get_spots_counts(args):
    """
    Aggregates all files containing spot counts files that match the pattern
    into an output csv file
    """
    labels_attrs = read_array_attrs(args.labels_container, args.labels_dataset)
    if args.voxel_spacing:
        voxel_spacing = args.voxel_spacing[::-1]
    else:
        # get voxel spacing from input image attributes
        voxel_spacing = get_spatial_voxel_spacing(labels_attrs)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
            voxel_spacing /= expansion
    else:
        voxel_spacing = (1,) * 3

    logger.info(f'Image voxel spacing: {voxel_spacing}')

    fx = sorted(glob(args.spots_pattern))

    labels_zarr = open_array(args.labels_container, args.labels_dataset)
    labels = read_zarr_block(labels_zarr, labels_attrs, args.spots_timeindex, args.spots_channel, None)
    label_ids = np.unique(labels[labels != 0])
    z, y, x = labels.shape[-3:]
    logger.info(f'Found {len(label_ids)} labels - labels shape: {labels.shape}')

    count = pd.DataFrame(np.empty([len(label_ids), 0]), index=label_ids)

    for f in fx:
        logger.info(f'Reading {f}')
        r = os.path.basename(f).split('/')[-1]
        r = r.split('.')[0]
        spot = np.loadtxt(f, delimiter=',')
        n = len(spot)

        # Convert from micrometer space to the voxel space of the segmented image
        spots_coords = spot[:, :3]/voxel_spacing
        df = pd.DataFrame(np.zeros([len(label_ids), 1]), index=label_ids, columns=['count'])

        for i in range(0, n):
            if np.any(np.isnan(spot[i,:3])):
                logger.info(f'NaN found in {f} line# {i+1}')
            else:
                if np.any(spot[i,:3]<0):
                    logger.info(f'Point outside of fixed image found in {f} line# {i+1} -> {spots_coords[i]}, {spot[i]}')
                else:
                    try:
                        # if all non-rounded coord are valid values (none is NaN)
                        coord = np.minimum(spots_coords[i], [x, y, z])
                        spot_label = _get_spot_label(labels, args.spots_timeindex, args.spots_channel, coord)
                        if spot_label > 0 and spot_label <= len(label_ids):
                            # increment counter
                            df.loc[spot_label, 'count'] = df.loc[spot_label, 'count'] + 1
                    except Exception as e:
                        logger.exception(f'Unexpected error in {f} line# {i+1}:')
                        traceback.print_exception(e)

        count.loc[:, r] = df.to_numpy()

    filtered_count = count[(count.iloc[:, -3:] != 0).any(axis=1)]

    logger.info(f'Writing {args.output}')
    filtered_count.to_csv(args.output, index_label='Label')


def _get_spot_label(labels, timeindex, channel, xyz_coord):
    zyx_coord = xyz_coord[::-1]
    max_coord = np.round(zyx_coord).astype(int)
    min_coord = np.maximum(np.floor(zyx_coord-1).astype(int), [0, 0, 0])
    crange = ()
    try:
        label_ndims = labels.ndim
        time_coord = (timeindex,) if timeindex is not None and label_ndims > 3 else ()
        channel_coord = (channel,) if channel is not None and label_ndims > 3 else ()
        crange = time_coord + channel_coord + tuple([slice(start, stop) 
                                                     if start != stop else start 
                                                     for start, stop in 
                                                     zip(min_coord, max_coord)])
        return np.max(labels[crange])
    except Exception as e:
        logger.exception(f'Error retrieving label at {xyz_coord} using {crange}: {e}')
        return 0


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)
    logger.info(f'Invoked spots counts with: {args}')

    # run post processing
    _get_spots_counts(args)


if __name__ == '__main__':
    _main()
