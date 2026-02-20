import argparse
import logging
import numpy as np
import os
import pandas as pd
import zarr

from glob import glob
from zarr_tools.ngff.ngff_utils import get_spatial_dataset_voxel_spacing


from .cli import floattuple
from .io_utils.read_utils import open_array, read_array_attrs
from .utils.configure_logging import configure_logging


logger:logging.Logger


def _inttuple(arg):
    if arg is not None and arg.strip():
        return tuple([int(d) for d in arg.split(',')])
    else:
        return ()


def _define_args():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--image-container',
                             dest='image_container',
                             type=str,
                             help = "image container")
    args_parser.add_argument('--image-subpath', '--image-dataset',
                             dest='image_dataset',
                             type=str,
                             help = "image subpath")

    args_parser.add_argument('--labels-container',
                             dest='labels_container',
                             type=str,
                             required=True,
                             help = "path to the labels container")
    args_parser.add_argument('--labels-subpath', '--labels-dataset',
                             dest='labels_dataset',
                             type=str,
                             required=False,
                             help = "path to the labels container")

    args_parser.add_argument('--labels-voxel-spacing', '--labels_voxel_spacing',
                             dest='labels_voxel_spacing',
                             type=floattuple,
                             metavar="SX,SY,SZ",
                             help = "Labels voxel spacing")
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
                             dest='labels_timeindex',
                             type=int,
                             help = "Time index from the labels OME ZARR used for spots counting")

    args_parser.add_argument('--channel',
                             dest='labels_channel',
                             type=int,
                             help = "Channel index from the labels OME ZARR used for spots counting")

    args_parser.add_argument('--processing-blocksize',
                             dest='processing_blocksize',
                             metavar='X,Y,Z',
                             type=_inttuple,
                             help='Block processing size')

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
    if args.labels_voxel_spacing:
        labels_voxel_spacing = np.array(args.labels_voxel_spacing)
    else:
        labels_attrs = read_array_attrs(args.labels_container, args.labels_dataset)
        # get label voxel spacing from labels image attributes
        labels_voxel_spacing = get_spatial_dataset_voxel_spacing(labels_attrs, args.labels_dataset)

    if labels_voxel_spacing is None:
        # the segmentation image does not have information about the voxel spacing
        # try to get that info from the image
        image_attrs = read_array_attrs(args.image_container, args.image_dataset)
        # get label voxel spacing from labels image attributes
        labels_voxel_spacing = get_spatial_dataset_voxel_spacing(image_attrs, args.image_dataset)

    if labels_voxel_spacing is not None:
        if args.expansion_factor > 0:
            expansion = args.expansion_factor
            labels_voxel_spacing /= expansion
    else:
        labels_voxel_spacing = (1,) * 3

    logger.info(f'Labels voxel spacing at {args.labels_dataset}: {labels_voxel_spacing}')

    processing_blocksize = args.processing_blocksize[::-1] if args.processing_blocksize else None

    fx = sorted(glob(args.spots_pattern))

    spots_per_file = {}
    for f in fx:
        logger.info(f'Reading {f}')
        r = os.path.basename(f).split('/')[-1]
        r = r.split('.')[0]
        df = pd.read_csv(f, header='infer')
        if df.columns.dtype == object:
            # file has a text header — column names are strings
            spots = df.values
        else:
            # no header — first row was data, re-read without header
            spots = pd.read_csv(f, header=None).values

        # Convert from micrometer space to the voxel space of the segmented image
        # CSV columns are x,y,z — convert to z,y,x and scale to voxel space
        spots_zyx = spots[:, :3][:, ::-1] / labels_voxel_spacing
        # round the zyx coordinates to the nearest int
        spots_per_file[f] = np.round(spots_zyx).astype(int)

    spot_counts = pd.DataFrame()
    # open labels image
    labels_zarr = open_array(args.labels_container, args.labels_dataset)
    _blockwise_spot_count(labels_zarr,
                          args.labels_timeindex,
                          args.labels_channel,
                          processing_blocksize,
                          spots_per_file,
                          labels_voxel_spacing,
                          spot_counts)

    # drop rows where all file columns are zero
    if not spot_counts.empty:
        spot_counts = spot_counts[(spot_counts != 0).any(axis=1)]
        spot_counts.index.name = 'Label'
        spot_counts = spot_counts.sort_index()

    logger.info(f'Writing {args.output}')
    spot_counts.to_csv(args.output, index_label='Label')


def _blockwise_spot_count(labels_zarr:zarr.Array,
                          timeindex:int|None,
                          labels_channel:int|None,
                          blocksize,
                          spots_files, # mapping of file -> ndarray[int]
                          voxel_spacing,
                          counts):
    """Count spots per label by reading one labels block at a time.

    Parameters
    ----------
    labels_zarr : zarr.Array
    labels_attrs : dict
    timeindex : int or None
    labels_channel : int or None
    blocksize : tuple or None
        Spatial block size in ZYX. If None, use the full volume as a single block.
    spots_files : a dict containing files with spot positions read from the file
    counts : dataframe for spot counts result

    Returns
    -------
    counts : pd.Dataframe
        Mapping of label_id -> spot count.
    """
    spatial_shape = np.array(labels_zarr.shape[-3:])
    blocksize = np.array(blocksize) if blocksize is not None else spatial_shape
    nblocks = np.ceil(spatial_shape / blocksize).astype(int)
    logger.info(f'Split {spatial_shape} labels image into {nblocks} blocks of size {blocksize}')
    timeindex_and_channel = []
    if timeindex is not None:
        timeindex_and_channel.append(timeindex)
    if labels_channel is not None:
        timeindex_and_channel.append(labels_channel)

    for block_index in np.ndindex(*nblocks):
        start = blocksize * np.array(block_index)
        stop = np.minimum(spatial_shape, start + blocksize)
        block_coords = tuple(timeindex_and_channel + [slice(int(s), int(e)) for s, e in zip(start, stop)])
        logger.info(f'Reading labels block {block_index} / {tuple(nblocks)}: {block_coords}')
        block_labels = labels_zarr[block_coords]
        block_label_ids = np.unique(block_labels[block_labels != 0])
        logger.info(f'Block {block_index} ({block_labels.shape}) found {len(block_label_ids)} labels')

        for f, spots_zyx in spots_files.items():
            r = os.path.splitext(os.path.basename(f))[0]
            if r not in counts.columns:
                counts[r] = 0

            # filter spots that fall within this block's [start, stop) range
            in_block = np.all((spots_zyx >= start) & (spots_zyx < stop), axis=1)
            block_spots_zyx = spots_zyx[in_block]
            logger.info(f'Block {block_index} file {r}: {len(block_spots_zyx)} spots')

            for spot_zyx in block_spots_zyx:
                if np.any(np.isnan(spot_zyx)):
                    continue
                local_idx = spot_zyx - start
                if np.any(local_idx < 0) or np.any(local_idx > block_labels.shape):
                    logger.warning((
                        f'Block {block_index} at {block_coords} '
                        f'unexpected out of bounds for {spot_zyx} ({spot_zyx * voxel_spacing}) -> {local_idx} '
                    ))
                    continue
                try:
                    label = int(block_labels[tuple(local_idx)])
                    if label > 0:
                        if label not in counts.index:
                            counts.loc[label] = 0
                        counts.loc[label, r] += 1
                except Exception as e:
                    logger.exception(f'Error looking up label at {spot_zyx}: {e}')

    return counts



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
