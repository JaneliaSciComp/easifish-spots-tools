import functools
import logging
import numpy as np
import zarr

import fishspot.filter as fs_filter
import fishspot.psf as fs_psf
import fishspot.detect as fs_detect

from itertools import product
from scipy.ndimage import gaussian_filter
from typing import List


logger = logging.getLogger(__name__)


def distributed_spot_detection(
    image_data:zarr.Array,
    timeindex:int|None,
    channels:List[int]|None,
    excluded_channels:set[int]|None,
    blocksize,
    dask_client,
    white_tophat_args:dict={},
    psf_estimation_args:dict={},
    deconvolution_args:dict={},
    spot_detection_args:dict={},
    gaussian_sigma:float|None=None,
    intensity_threshold=None,
    intensity_threshold_minimum=0,
    mask=None,
    psf=None,
    psf_retries=3,
):
    # set white_tophat defaults
    if 'radius' not in white_tophat_args:
        white_tophat_args['radius'] = 4

    # set psf estimation defaults
    if 'radius' not in psf_estimation_args:
        psf_estimation_args['radius'] = 9

    # set spot detection defaults
    if 'min_radius' not in spot_detection_args:
        spot_detection_args['min_radius'] = 1
    if 'max_radius' not in spot_detection_args:
        spot_detection_args['max_radius'] = 6

    # compute overlap depth
    all_radii = [white_tophat_args['radius'],
                 psf_estimation_args['radius'],
                 spot_detection_args['max_radius'],]
    overlap = int(2*max(np.max(x) for x in all_radii))

    # don't detect spots in the overlap region
    if 'exclude_border' not in spot_detection_args:
        spot_detection_args['exclude_border'] = overlap

    spatial_shape = image_data.shape[-3:]
    nspatial_dims = len(spatial_shape)

    # get the timeindexes to process
    # a timeindex value of -1 means that the timeindex dimension is not present
    if timeindex is None:
        spots_timepoints = tuple(range(image_data.shape[-nspatial_dims-2])) if len(image_data.shape) > nspatial_dims+1 else (-1,)
    else:
        spots_timepoints = (timeindex,)

    # get the channels to process
    # a channel value of -1 means that the channel dimension is not present
    if channels is None or channels == [] or channels == ():
        spots_channels = tuple(range(image_data.shape[-nspatial_dims-1])) if len(image_data.shape) > nspatial_dims else (-1,)
    elif isinstance(channels, tuple) or isinstance(channels, list):
        spots_channels = channels
    elif isinstance(channels, int):
        spots_channels = (channels,)
    else:
        spots_channels = (-1,)

    if excluded_channels is None:
        excluded_spots_channels = ()
    elif isinstance(excluded_channels, int):
        excluded_spots_channels = (excluded_channels, )
    elif isinstance(excluded_channels, tuple) or isinstance(excluded_channels, list):
        excluded_spots_channels = excluded_channels
    else:
        excluded_spots_channels = ()
    

    # compute mask to array ratio
    if mask is not None:
        ratio = np.array(mask.shape) / spatial_shape
        stride = np.round(blocksize * ratio).astype(int)

    # compute number of blocks
    nblocks = np.ceil(np.array(spatial_shape) / blocksize).astype(int)

    # determine coords for blocking
    overlap_coords, core_coords, psfs = [], [], []
    for tp in spots_timepoints:
        for ch in spots_channels:

            if ch in excluded_spots_channels:
                continue

            for (i, j, k) in product(*[range(x) for x in nblocks]):
                # determine if block is in foreground
                if mask is not None:
                    mo = stride * (i, j, k)
                    mask_slice = tuple(slice(x, x+y) for x, y in zip(mo, stride))
                    if not np.any(mask[mask_slice]):
                        continue

                # create overlap coords and append to list
                extended_start = np.array(blocksize) * (i, j, k) - overlap
                extended_stop = extended_start + blocksize + 2 * overlap
                extended_start = np.maximum(0, extended_start)
                extended_stop = np.minimum(spatial_shape, extended_stop)
                # create core coords
                start = np.array(blocksize) * (i, j, k)
                stop = start + blocksize
                stop = np.minimum(spatial_shape, stop)

                block_coords, extended_coords = [], []
                if tp != -1:
                    block_coords.append(tp)
                    extended_coords.append(tp)
                if ch != -1:
                    block_coords.append(ch)
                    extended_coords.append(ch)

                block_coords.extend([slice(x, y) for x, y in zip(start, stop)])
                extended_coords.extend([slice(x, y) for x, y in zip(extended_start, extended_stop)])

                # add coordinate slices to list
                core_coords.append(block_coords)
                overlap_coords.append(tuple(extended_coords))
                psfs.append(psf)

    logger.info(f'Partition an {spatial_shape} volume into {len(core_coords)} blocks for spot detection')
    blocks = dask_client.map(_read_block, core_coords, overlap_coords, array=image_data)

    # submit all alignments to cluster
    detect_block_spots = functools.partial(
        _detect_block_spots,
        psf_estimation_args=psf_estimation_args,
        white_tophat_args=white_tophat_args,
        deconvolution_args=deconvolution_args,
        spot_detection_args=spot_detection_args,
        gaussian_sigma=gaussian_sigma,
        intensity_threshold=intensity_threshold,
        intensity_threshold_minimum=intensity_threshold_minimum,
        psf_retries=psf_retries,
    )
    spots_and_psfs = dask_client.gather(
        dask_client.map(detect_block_spots, blocks, psfs)
    )
    # reformat to single array of spots and single psf
    spots, psfs = [], []
    for x, y in spots_and_psfs:
        spots.append(x)
        if y is not None:
            psfs.append(y)
    if len(spots) > 0:
        spots = np.vstack(spots)
    if len(psfs) > 0:
        psf = np.mean(psfs, axis=0)
    else:
        psf = None

    # filter with foreground mask
    if mask is not None:
        spots = fs_filter.apply_foreground_mask(
            spots, mask, ratio,
        )

    # return results
    return spots, psf


def _read_block(core_coords, overlap_coords, array=[]):
    return array[overlap_coords], core_coords, overlap_coords


# pipeline to run on each block
def _detect_block_spots(block_with_coords, psf,
                        psf_estimation_args={},
                        white_tophat_args={},
                        deconvolution_args={},
                        spot_detection_args={},
                        gaussian_sigma=None,
                        intensity_threshold=None,
                        intensity_threshold_minimum=0,
                        psf_retries=3):
    original_block = block_with_coords[0]
    core_coords = block_with_coords[1]
    overlap_coords = block_with_coords[2]

    # make a copy of the block
    block = np.copy(original_block)

    if len(overlap_coords) > len(block.shape):
        ch = overlap_coords[-len(block.shape)-1] + 1
    else:
        ch = 1
    if len(overlap_coords) > len(block.shape) + 1:
        timeindex = overlap_coords[-len(block.shape)-2] + 1
    else:
        timeindex = 1
    
    logger.info(f'Detect spots for {core_coords} ({overlap_coords}) block of size {block.shape}')

    # load data, background subtract, deconvolve, detect blobs
    wth_filtered_block = fs_filter.white_tophat(block, **white_tophat_args)

    # optional smoothing, Note: should only be with extremely small sigmas
    if gaussian_sigma is not None:
        processed_block = gaussian_filter(wth_filtered_block, gaussian_sigma)
    else:
        processed_block = wth_filtered_block

    if psf is None:
        # automated psf estimation with error handling
        for i in range(psf_retries):
            try:
                psf = fs_psf.estimate_psf(wth_filtered_block, **psf_estimation_args)
            except ValueError as ve:
                if 'inlier_threshold' not in psf_estimation_args:
                    psf_estimation_args['inlier_threshold'] = 0.9

                psf_estimation_args['inlier_threshold'] -= 0.1
            else:
                break

    if psf is not None:
        logger.info(f'PSF {psf.shape} found for {core_coords} ({overlap_coords}) block')
        decon = fs_filter.rl_decon(processed_block, psf, **deconvolution_args)
    else:
        logger.info(f'No PSF could be estimated for {block.shape} block at {core_coords} ({overlap_coords})')
        decon = processed_block

    # final spot detection
    spots = fs_detect.detect_spots_log(decon, **spot_detection_args)
    logger.info(f'Initial spots array shape: {spots.shape}')

    if spots.shape[0] == 0:
        # if no spots are found, ensure consistent format - z,y,x,intensity,sigma_x,sigma_y,sigma_z
        return np.zeros((0, len(block.shape) + 6)), psf
    else:
        # remove spots found in the overlap region
        core_origin = [x.start-y.start for x, y in zip(core_coords[-3:], overlap_coords[-3:])]
        span = [x.stop-x.start for x in core_coords[-3:]]
        spots = fs_filter.filter_by_range(spots, core_origin, span)
        logger.info(f'Spots array shape after overlap removal: {spots.shape}')

        # get an intensity threshold
        if intensity_threshold is None:
            intensity_threshold = fs_filter.maximum_deviation_threshold(original_block, winsorize=(1, 99.995))
            intensity_threshold = max(intensity_threshold, intensity_threshold_minimum)
        logger.info(f'Using intensity threshold: {intensity_threshold}')

        # spots shape -> (n_spots, 6) [[x, y, z, sigma_x, sigma_y, sigma_z]]
        # append image intensities
        nspots = spots.shape[0]
        spot_coords = spots[:, :3].astype(int)
        intensities = block[spot_coords[:, 0], spot_coords[:, 1], spot_coords[:, 2]]

        ti_arr = np.repeat(np.array([[timeindex]]), nspots, axis=0)
        ch_arr = np.repeat(np.array([[ch]]), nspots, axis=0)
        spots = np.concatenate((spots[:,:3],
                                ti_arr,
                                ch_arr,
                                intensities[..., None],
                                spots[:,3:]), axis=1)
        # filter by intensity (intensity_dim == 5)
        spots = spots[ spots[..., 5] > intensity_threshold ]

        # adjust for block origin
        origin = np.array([x.start for x in overlap_coords[-3:]])
        spots[:, :3] = spots[:, :3] + origin
        logger.info(f'Block {core_coords} ({overlap_coords}) -> found: {spots.shape} spots')
        return spots, psf
