import argparse
import logging
import numpy as np
import time

from dask.distributed import (Client, LocalCluster)
from pathlib import Path

from zarr_tools.ngff.ngff_utils import get_spatial_voxel_spacing

from .cli import floattuple, inttuple
from .io_utils.read_utils import open_array, read_array_attrs
from .spot_detection.configure_fishspots import get_fishspots_config
from .spot_detection.distributed_spot_detection import distributed_spot_detection
from .utils.configure_dask import (ConfigureWorkerPlugin, load_dask_config)
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
    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             help='Image voxel spacing')
    args_parser.add_argument('--timeindex',
                             type=int,
                             default=None,
                             help='Time index to process (if applicable)')
    args_parser.add_argument('--channels', '--included-channels',
                             dest='channels',
                             type=int,
                             nargs='+',
                             default=[],
                             help='List of channel indices to process')
    args_parser.add_argument('--excluded-channels',
                             dest='excluded_channels',
                             type=int,
                             nargs='+',
                             default=[],
                             help='List of channel indices to process')
    args_parser.add_argument('--output',
                             type=str,
                             required=True,
                             help='Path to the output file')
    args_parser.add_argument('--apply-voxel-spacing', '--apply_voxel_spacing',
                             dest='apply_voxel_spacing',
                             action='store_true',
                             default=False,
                             help="Apply voxel spacing")

    args_parser.add_argument('--dask-scheduler',
                             dest='dask_scheduler',
                             type=str,
                             default=None,
                             help='Run with distributed scheduler')
    args_parser.add_argument('--dask-config', dest='dask_config',
                             type=str, default=None,
                             help='YAML file containing dask configuration')
    args_parser.add_argument('--local-dask-workers', '--local_dask_workers',
                             dest='local_dask_workers',
                             type=int,
                             help='Number of workers when using a local cluster')
    args_parser.add_argument('--worker-cpus',
                             dest='worker_cpus',
                             type=int, default=0,
                             help='Number of cpus allocated to a dask worker')

    args_parser.add_argument('--blocksize',
                             type=inttuple,
                             default=(),
                             help='Block size as [x,y,z] size')

    args_parser.add_argument('--fishspots-config', '--fishspots_config',
                             dest='fishspots_config',
                             type=str,
                             help='Fishspots config yaml file')
    args_parser.add_argument('--psf-file', '--psf_file',
                             dest='psf_file',
                             type=str,
                             help='numpy file that contains the PSF')
    args_parser.add_argument('--psf-retries', '--psf_retries',
                             dest='psf_retries',
                             type=int,
                             default=3,
                             help='PSF retries')
    args_parser.add_argument('--gaussian-sigma',
                             dest='gaussian_sigma',
                             type=float,
                             help='Gaussian sigma')
    args_parser.add_argument('--intensity-threshold', '--intensity_threshold',
                             dest='intensity_threshold',
                             type=int,
                             help='Intensity threshold for spot detection')
    args_parser.add_argument('--intensity-threshold-minimum',
                             dest='intensity_threshold_minimum',
                             type=int,
                             default=0,
                             help='Intensity threshold minimum for spot detection')

    args_parser.add_argument('--logging-config', dest='logging_config',
                             type=str,
                             help='python log file configuration')
    args_parser.add_argument('--verbose',
                             action='store_true',
                             help="save debug level information to the log")

    return args_parser


def _write_spots(spots, csvfilename):
    # x,y,z -> z,y,x
    spots[:, :3] = spots[:, :3][:,::-1]

    header = 'x,y,z,t,c,intensity,sx,sy,sz'
    fmt = ['%.4f', '%.4f', '%.4f', '%d', '%d', '%.4f', '%.4f', '%.4f', '%.4f']

    logger.info(f'Write {len(spots)} spots to {csvfilename}')
    output_path = Path(csvfilename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(csvfilename, spots, delimiter=',', header=header, fmt=fmt)


def _main():
    # parse CLI args
    args_parser = _define_args()
    args = args_parser.parse_args()

    # prepare logging
    global logger
    logger = configure_logging(args.logging_config, args.verbose)
    logger.info(f'Invoked FishSpots with: {args}')

    load_dask_config(args.dask_config)

    if args.dask_scheduler:
        cluster_client = Client(address=args.dask_scheduler)
    else:
        cluster_client = Client(LocalCluster(n_workers=args.local_dask_workers,
                                             threads_per_worker=args.worker_cpus))

    worker_config = ConfigureWorkerPlugin(args.logging_config,
                                          args.verbose,
                                          worker_cpus=args.worker_cpus)
    cluster_client.register_plugin(worker_config, name='WorkerConfig')

    input_image_attrs = read_array_attrs(args.input, args.input_subpath)
    input_image_shape = input_image_attrs['array_shape']

    if args.voxel_spacing is not None:
        # voxel spacing is specified in the command line, so use this value
        voxel_spacing = args.voxel_spacing[::-1] # this is specified as XYZ and we want it as ZYX
    else:
        voxel_spacing = get_spatial_voxel_spacing(input_image_attrs)


    if args.blocksize:
        # convert the x,y,z input block size to z,y,x
        processing_blocksize = args.blocksize[::-1]
    else:
        processing_blocksize = input_image_shape[-3:]

    input_image_array = open_array(input_image_attrs['array_storepath'], input_image_attrs['array_subpath'])

    fishspots_config = get_fishspots_config(args.fishspots_config)
    white_tophat_args=fishspots_config.get('white_tophat_args', {})
    psf_estimation_args=fishspots_config.get('psf_estimation_args', {})
    deconvolution_args=fishspots_config.get('deconvolution_args', {})
    spot_detection_args=fishspots_config.get('spot_detection_args', {})

    psf_file = Path(args.psf_file) if args.psf_file else None
    if psf_file is not None and psf_file.exists() and psf_file.is_file():
        psf = np.load(psf_file)
    else:
        psf = None

    start_time = time.time()

    spots, _ = distributed_spot_detection(
        input_image_array,
        args.timeindex,
        args.channels,
        set(args.excluded_channels) if args.excluded_channels else set(),
        processing_blocksize,
        cluster_client,
        white_tophat_args=white_tophat_args,
        psf_estimation_args=psf_estimation_args,
        deconvolution_args=deconvolution_args,
        spot_detection_args=spot_detection_args,
        gaussian_sigma=args.gaussian_sigma,
        intensity_threshold=args.intensity_threshold,
        intensity_threshold_minimum=args.intensity_threshold_minimum,
        psf=psf,
        psf_retries=args.psf_retries,
    )

    elapsed_time = time.time() - start_time
    logger.info(f'Distributed spot detection completed in {elapsed_time:.2f} seconds')

    if args.apply_voxel_spacing:
        logger.info(f'Apply voxel spacing: {voxel_spacing}')
        spots[:, :3] = spots[:, :3] * voxel_spacing
    
    _write_spots(spots, args.output)

if __name__ == '__main__':
    _main()
