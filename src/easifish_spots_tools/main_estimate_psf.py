import argparse
import fishspot.psf as fs_psf
import logging
import numpy as np

from pathlib import Path

from .cli import floattuple, inttuple
from .io_utils.read_utils import open_array, read_array_attrs
from .spot_detection.configure_fishspots import get_fishspots_config
from .utils.configure_logging import configure_logging


logger:logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--input', 
                             type=str,
                             required=True,
                             help='Path to the input container')
    args_parser.add_argument('--input-subpath', '--input_subpath',
                             type=str,
                             default=None,
                             help='Optional dataset subpath within the input')
    args_parser.add_argument('--timeindex',
                             type=int,
                             default=None,
                             help='Time index to process (if applicable)')
    args_parser.add_argument('--channel',
                             dest='channel',
                             type=int,
                             default=None,
                             help='Channel used for estimating psf')
    args_parser.add_argument('--crop-offset', '--crop_offset',
                             dest='crop_offset',
                             type=inttuple,
                             metavar='X,Y,Z',
                             help='Crop offset')
    args_parser.add_argument('--crop-size', '--crop_size',
                             dest='crop_size',
                             type=inttuple,
                             default=(512,512,512),
                             metavar='SX,SY,SZ',
                             help='Crop size')

    args_parser.add_argument('--psf-crop-file',
                             dest='psf_crop_file',
                             type=str,
                             help='Path to save the crop used for estimating the PSF')
    args_parser.add_argument('--psf-file',
                             dest='psf_file',
                             type=str,
                             required=True,
                             help='Path to the output file')

    args_parser.add_argument('--fishspots-config', '--fishspots_config',
                             dest='fishspots_config',
                             type=str,
                             help='Fishspots config yaml file')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='python log file configuration')
    args_parser.add_argument('--verbose',
                             action='store_true',
                             help="save debug level information to the log")

    return args_parser


def _main():
    # parse CLI args
    args_parser = _define_args()
    args = args_parser.parse_args()

    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)
    logger.info(f'Invoked FishSpots with: {args}')

    input_image_attrs = read_array_attrs(args.input, args.input_subpath)
    spatial_image_shape = input_image_attrs['array_shape'][-3:]

    input_image_array = open_array(input_image_attrs['array_storepath'], input_image_attrs['array_subpath'])

    fishspots_config = get_fishspots_config(args.fishspots_config)
    psf_estimation_args = fishspots_config.get('psf_estimation_args', {})

    # read the image block using args.crop_offset and args.crop_size
    # crop_offset and crop_size are in x,y,z order — reverse to z,y,x for array indexing
    crop_offset_zyx = args.crop_offset[::-1] if args.crop_offset else (0, 0, 0)
    crop_size_zyx = args.crop_size[::-1]
    spatial_slices = tuple(slice(o, min(o + s, dim_size))
                           for o, s, dim_size in zip(crop_offset_zyx, crop_size_zyx, spatial_image_shape))

    if args.timeindex is not None and args.channel is not None:
        index = (args.timeindex, args.channel, *spatial_slices)
    elif args.timeindex is not None:
        index = (args.timeindex, *spatial_slices)
    elif args.channel is not None:
        index = (args.channel, *spatial_slices)
    else:
        index = spatial_slices

    crop = np.copy(input_image_array[index])
    logger.info(f'Loaded crop of shape {crop.shape}')

    # if args.psf_crop_file is set save the crop as nrrd
    if args.psf_crop_file:
        import nrrd
        psf_crop_file = Path(args.psf_crop_file)
        psf_crop_file.parent.mkdir(parents=True, exist_ok=True)
        nrrd.write(str(psf_crop_file), crop.transpose(2, 1, 0))  # z,y,x -> x,y,z
        logger.info(f'Saved PSF crop to {psf_crop_file}')

    # call fishspot.psf.estimate_psf with psf_estimation_args for the crop
    logger.info('Estimating PSF...')
    psf = fs_psf.estimate_psf(crop, **psf_estimation_args)
    logger.info(f'Estimated PSF of shape {psf.shape}')

    psf_file = Path(args.psf_file)
    psf_file.parent.mkdir(parents=True, exist_ok=True)

    # save the computed PSF using the same logic used for reading the PSF in main_spot_extraction
    if psf_file.suffix == '.npy':
        np.save(psf_file, psf)
    elif psf_file.suffix == '.nrrd':
        import nrrd
        nrrd.write(str(psf_file), psf.transpose(2, 1, 0))  # z,y,x -> x,y,z
    else:
        raise ValueError(f'Unsupported PSF file format: {psf_file.suffix}')
    logger.info(f'Saved PSF to {psf_file}')

if __name__ == '__main__':
    _main()
