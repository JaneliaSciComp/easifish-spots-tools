import argparse
import logging
import numpy as np
import pandas as pd


from skimage.measure import regionprops
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


def _extract_spots_region_properties(args):
    image_attrs = read_array_attrs(args.image_container, args.image_dataset)
    image_zarr = open_array(args.image_container, args.image_dataset)
    logger.info(f'Opened {image_zarr.shape} image {args.image_container}:{args.image_dataset}')

    labels_attrs = read_array_attrs(args.labels_container, args.labels_dataset)
    labels_zarr = open_array(args.labels_container, args.labels_dataset)
    logger.info(f'Opened {labels_zarr.shape} labels {args.labels_container}:{args.labels_dataset}')

    if args.voxel_spacing:
        voxel_spacing = args.voxel_spacing[::-1]
    else:
        # get voxel spacing from input image attributes
        voxel_spacing = get_spatial_voxel_spacing(image_attrs)

    if voxel_spacing is not None:
        if args.expansion_factor > 0:
            voxel_spacing = [c / args.expansion_factor for c in voxel_spacing]
    else:
        voxel_spacing = (1,) * 3

    # read the entire image
    image = read_zarr_block(image_zarr, image_attrs, args.image_timeindex, args.image_channel, None)

    if ((args.bleeding_dataset is not None and
         args.dapi_dataset is not None and
         args.bleeding_dataset == args.image_dataset) or
        (args.bleeding_channel is not None and
         args.dapi_channel is not None and
         args.bleeding_channel == args.image_channel)):
        dapi_attrs = read_array_attrs(args.image_container, args.dapi_dataset)
        dapi_zarr = open_array(args.image_container, args.dapi_dataset)
        logger.info(f'Opened {dapi_zarr.shape} DAPI image {args.image_container}:{args.dapi_dataset}')
        # read the entire DAPI image
        dapi = read_zarr_block(dapi_zarr, dapi_attrs, args.image_timeindex, args.dapi_channel, None)
        lo = np.percentile(np.ndarray.flatten(dapi), 99.5)
        bg_dapi = np.percentile(np.ndarray.flatten(dapi[dapi != 0]), 1)
        bg_img = np.percentile(np.ndarray.flatten(image[image != 0]), 1)
        dapi_factor = np.median((image[dapi > lo] - bg_img) /
                                (dapi[dapi > lo] - bg_dapi))
        image = np.maximum(0, image - bg_img - dapi_factor * (dapi - bg_dapi)).astype('float32')
        logger.info(f'Corrected bleed dataset {args.image_dataset} {image.shape} image')
        logger.info(f'bleed_through: {dapi_factor}')
        logger.info(f'DAPI background: {bg_dapi}')
        logger.info(f'bleed_through channel background: {bg_img}')

    labels = read_zarr_block(labels_zarr, labels_attrs, args.labels_timeindex, args.labels_channel, None)
    logger.info(f'Extract regionprops from {labels.shape} labels and {image.shape} image')
    labels_stats = regionprops(labels, intensity_image=image, spacing=voxel_spacing)

    df = pd.DataFrame(data=np.empty([len(labels_stats), 3]),
                      columns=['roi', 'mean_intensity', 'area'],
                      dtype=object)

    for i in range(0, len(labels_stats)):
        df.loc[i, 'roi'] = labels_stats[i].label
        df.loc[i, 'mean_intensity'] = labels_stats[i].intensity_mean
        df.loc[i, 'area'] = labels_stats[i].area

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
