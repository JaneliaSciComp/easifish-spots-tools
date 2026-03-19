import argparse
import logging
import numpy as np
import pandas as pd
from zarr_tools.ngff.ngff_utils import get_spatial_voxel_spacing

from .cli import floattuple, inttuple
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
    args_parser.add_argument('--labels-timeindex',
                             dest='labels_timeindex',
                             type=int,
                             help = "labels time index")
    args_parser.add_argument('--labels-channel',
                             dest='labels_channel',
                             type=int,
                             help = "labels channel")

    args_parser.add_argument('--image-container',
                             dest='image_container',
                             type=str,
                             help = "image container")
    args_parser.add_argument('--image-subpath', '--image-dataset',
                             dest='image_dataset',
                             type=str,
                             help = "image subpath")
    args_parser.add_argument('--image-timeindex',
                             dest='image_timeindex',
                             type=int,
                             help = "image time index")
    args_parser.add_argument('--image-channel',
                             dest='image_channel',
                             type=int,
                             help = "image channel")

    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             metavar='X,Y,Z',
                             help = "Spatial voxel spacing as X,Y,Z")
    args_parser.add_argument('--expansion-factor', '--expansion_factor',
                             dest='expansion_factor',
                             type=float,
                             default=0.,
                             help='Sample expansion factor')

    args_parser.add_argument('--dapi-subpath', '--dapi-dataset',
                             dest='dapi_dataset',
                             type=str,
                             help = "DAPI image subpath")
    args_parser.add_argument('--dapi-channel',
                             dest='dapi_channel',
                             type=int,
                             help = "DAPI channel")
    args_parser.add_argument('--bleeding-subpath', '--bleeding-dataset',
                             dest='bleeding_dataset',
                             type=str,
                             help = "Bleeding image subpath")
    args_parser.add_argument('--bleeding-channel',
                             dest='bleeding_channel',
                             type=int,
                             help = "Bleeding channel")

    args_parser.add_argument('--processing-blocksize',
                             dest='processing_blocksize',
                             metavar='X,Y,Z',
                             type=inttuple,
                             help='Block processing size in X,Y,Z (reversed to ZYX internally)')

    args_parser.add_argument('-o','--output',
                             dest='output',
                             type=str,
                             required=True,
                             help = "output file in real coordinates")

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='python log file configuration')
    args_parser.add_argument('--verbose',
                             action='store_true',
                             help="save debug level information to the log")

    return args_parser


def _build_timeindex_prefix(zarr_array, timeindex, channel):
    """Build list of prefix indices for time and channel dimensions.

    Parameters
    ----------
    zarr_array : zarr.Array
    timeindex : int or None
    channel : int or None

    Returns
    -------
    list[int]
    """
    prefix = []
    if timeindex is not None:
        prefix.append(timeindex)
    if channel is not None:
        prefix.append(channel)
    if len(prefix) + 3 < zarr_array.ndim:
        logger.warning((
            f'Timeindex or channel have not been specified for the {zarr_array.ndim}D array '
            f'of shape {zarr_array.shape} '
            'so automatically we use the 0 hyperplane for missing dimensions'
        ))
        prefix = [0] * (zarr_array.ndim - 3)
    return prefix


def _reservoir_sample(samples, count, new_values, max_samples):
    """Update a reservoir sample with new values (Algorithm R, vectorized).

    Parameters
    ----------
    samples : list
        Current reservoir, modified in-place.
    count : int
        Number of elements seen before new_values.
    new_values : np.ndarray
        New values to incorporate (any shape; will be ravelled).
    max_samples : int
        Maximum reservoir size.

    Returns
    -------
    int
        Updated total element count.
    """
    new_values = np.asarray(new_values, dtype='float32').ravel()
    n = len(new_values)
    if n == 0:
        return count

    # Fill phase: top up the reservoir if not yet full
    fill_needed = max(0, max_samples - len(samples))
    if fill_needed > 0:
        fill_n = min(fill_needed, n)
        samples.extend(new_values[:fill_n].tolist())
        new_values = new_values[fill_n:]
        count += fill_n
        n = len(new_values)
        if n == 0:
            return count

    # Replace phase: for element i (0-indexed among new_values),
    # the overall index is count+1+i; replace a random position if it < max_samples.
    overall_counts = count + 1 + np.arange(n)
    rand_positions = (np.random.random(n) * overall_counts).astype(int)
    accept_mask = rand_positions < max_samples
    for pos, val in zip(rand_positions[accept_mask], new_values[accept_mask]):
        samples[int(pos)] = float(val)

    count += n
    return count


def _compute_bleed_correction_params(image_zarr, dapi_zarr,
                                     image_prefix, dapi_prefix,
                                     spatial_shape, blocksize,
                                     max_samples=5_000_000):
    """Compute bleed-through correction scalars via two block passes.

    Pass A samples non-zero DAPI and image voxels to estimate lo, bg_dapi, bg_img.
    Pass B collects corrected ratios where dapi > lo to estimate dapi_factor.

    Parameters
    ----------
    image_zarr, dapi_zarr : zarr.Array
    image_prefix, dapi_prefix : list[int]
        Prefix indices (time, channel) for each array.
    spatial_shape : np.ndarray  (Z, Y, X)
    blocksize : np.ndarray  (Z, Y, X)
    max_samples : int

    Returns
    -------
    bg_img, bg_dapi, lo, dapi_factor : float
    """
    nblocks = np.ceil(spatial_shape / blocksize).astype(int)
    logger.info(f'Bleed correction pass A over {tuple(nblocks)} blocks')

    dapi_samples, img_samples = [], []
    dapi_count = img_count = 0

    for block_index in np.ndindex(*nblocks):
        start = blocksize * np.array(block_index)
        stop = np.minimum(spatial_shape, start + blocksize)
        spatial_slices = [slice(int(s), int(e)) for s, e in zip(start, stop)]
        img_block = image_zarr[tuple(image_prefix + spatial_slices)].astype('float32')
        dapi_block = dapi_zarr[tuple(dapi_prefix + spatial_slices)].astype('float32')
        dapi_count = _reservoir_sample(dapi_samples, dapi_count,
                                       dapi_block[dapi_block != 0], max_samples)
        img_count = _reservoir_sample(img_samples, img_count,
                                      img_block[img_block != 0], max_samples)

    dapi_arr = np.array(dapi_samples, dtype='float32')
    img_arr = np.array(img_samples, dtype='float32')
    lo = float(np.percentile(dapi_arr, 99.5)) if len(dapi_arr) > 0 else 0.0
    bg_dapi = float(np.percentile(dapi_arr, 1)) if len(dapi_arr) > 0 else 0.0
    bg_img = float(np.percentile(img_arr, 1)) if len(img_arr) > 0 else 0.0

    logger.info(f'Bleed correction pass B over {tuple(nblocks)} blocks (lo={lo})')
    ratio_samples = []
    ratio_count = 0

    for block_index in np.ndindex(*nblocks):
        start = blocksize * np.array(block_index)
        stop = np.minimum(spatial_shape, start + blocksize)
        spatial_slices = [slice(int(s), int(e)) for s, e in zip(start, stop)]
        img_block = image_zarr[tuple(image_prefix + spatial_slices)].astype('float32')
        dapi_block = dapi_zarr[tuple(dapi_prefix + spatial_slices)].astype('float32')
        mask = dapi_block > lo
        if mask.any():
            ratios = (img_block[mask] - bg_img) / (dapi_block[mask] - bg_dapi)
            ratio_count = _reservoir_sample(ratio_samples, ratio_count, ratios, max_samples)

    dapi_factor = float(np.median(ratio_samples)) if ratio_samples else 0.0
    return bg_img, bg_dapi, lo, dapi_factor


def _blockwise_region_props(labels_zarr, image_zarr, dapi_zarr,
                             labels_prefix, image_prefix, dapi_prefix,
                             blocksize, bleed_params):
    """Accumulate per-label sum_intensity and pixel_count over spatial blocks.

    Parameters
    ----------
    labels_zarr, image_zarr : zarr.Array
    dapi_zarr : zarr.Array or None
    labels_prefix, image_prefix, dapi_prefix : list[int]
    blocksize : np.ndarray  (Z, Y, X)
    bleed_params : tuple(bg_img, bg_dapi, lo, dapi_factor) or None
    Returns
    -------
    dict[int, [float, int]]
        label_id -> [sum_intensity, pixel_count]
    """
    spatial_shape = np.array(labels_zarr.shape[-3:])
    nblocks = np.ceil(spatial_shape / blocksize).astype(int)
    logger.info(f'Split {spatial_shape} labels image into {nblocks} blocks of size {blocksize}')

    accumulator = {}

    for block_index in np.ndindex(*nblocks):
        start = blocksize * np.array(block_index)
        stop = np.minimum(spatial_shape, start + blocksize)
        spatial_slices = [slice(int(s), int(e)) for s, e in zip(start, stop)]
        labels_coords = tuple(labels_prefix + spatial_slices)
        image_coords = tuple(image_prefix + spatial_slices)
        logger.info(f'Reading block {block_index} / {tuple(nblocks)}: {labels_coords}')

        labels_block = labels_zarr[labels_coords]
        image_block = image_zarr[image_coords]

        if bleed_params is not None:
            bg_img, bg_dapi, _, dapi_factor = bleed_params
            dapi_block = dapi_zarr[tuple(dapi_prefix + spatial_slices)].astype('float32')
            image_block = np.maximum(
                0, image_block.astype('float32') - bg_img - dapi_factor * (dapi_block - bg_dapi)
            )

        unique_labels = np.unique(labels_block)
        unique_labels = unique_labels[unique_labels != 0]
        logger.info(f'Block {block_index} ({labels_block.shape}) found {len(unique_labels)} labels')

        for label in unique_labels:
            mask = labels_block == label
            if label not in accumulator:
                accumulator[label] = [0, 0]
            accumulator[label][0] += image_block[mask].astype('float32').sum()
            accumulator[label][1] += mask.sum()

    return accumulator


def _extract_spots_region_properties(args):
    image_attrs = read_array_attrs(args.image_container, args.image_dataset)
    image_zarr = open_array(args.image_container, args.image_dataset)
    logger.info(f'Opened {image_zarr.shape} image {args.image_container}:{args.image_dataset}')

    labels_zarr = open_array(args.labels_container, args.labels_dataset)
    logger.info(f'Opened {labels_zarr.shape} labels {args.labels_container}:{args.labels_dataset}')

    if args.voxel_spacing:
        voxel_spacing = args.voxel_spacing[::-1]
    else:
        voxel_spacing = get_spatial_voxel_spacing(image_attrs)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            voxel_spacing = [c / args.expansion_factor for c in voxel_spacing]
    else:
        voxel_spacing = (1,) * 3

    spatial_shape = np.array(labels_zarr.shape[-3:])
    blocksize = (np.array(args.processing_blocksize[::-1])
                 if args.processing_blocksize else spatial_shape)

    labels_prefix = _build_timeindex_prefix(labels_zarr, args.labels_timeindex, args.labels_channel)
    image_prefix = _build_timeindex_prefix(image_zarr, args.image_timeindex, args.image_channel)

    bleed_params = None
    dapi_zarr = None
    dapi_prefix = []

    if ((args.bleeding_dataset is not None and
         args.dapi_dataset is not None and
         args.bleeding_dataset == args.image_dataset) or
        (args.bleeding_channel is not None and
         args.dapi_channel is not None and
         args.bleeding_channel == args.image_channel)):
        dapi_zarr = open_array(args.image_container, args.dapi_dataset)
        logger.info(f'Opened {dapi_zarr.shape} DAPI image {args.image_container}:{args.dapi_dataset}')
        dapi_prefix = _build_timeindex_prefix(dapi_zarr, args.image_timeindex, args.dapi_channel)
        bleed_params = _compute_bleed_correction_params(
            image_zarr, dapi_zarr,
            image_prefix, dapi_prefix,
            spatial_shape, blocksize,
        )
        bg_img, bg_dapi, _, dapi_factor = bleed_params
        logger.info(f'bleed_through: {dapi_factor}')
        logger.info(f'DAPI background: {bg_dapi}')
        logger.info(f'bleed_through channel background: {bg_img}')

    accumulator = _blockwise_region_props(
        labels_zarr, image_zarr, dapi_zarr,
        labels_prefix, image_prefix, dapi_prefix,
        blocksize, bleed_params,
    )

    voxel_volume = np.prod(voxel_spacing)
    rows = [(label, s / n, n * voxel_volume) for label, (s, n) in accumulator.items()]
    df = pd.DataFrame(rows, columns=['roi', 'mean_intensity', 'area'])
    df.sort_values('roi', inplace=True)

    logger.info(f'Writing {args.output}')
    df.to_csv(args.output, index=False)


def _main():
    args_parser = _define_args()
    args = args_parser.parse_args()

    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)
    logger.info(f'Invoked FishSpots with: {args}')

    # run post processing
    _extract_spots_region_properties(args)


if __name__ == '__main__':
    _main()
