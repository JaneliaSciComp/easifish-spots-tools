import argparse
import logging
import os
import numpy as np

from pathlib import Path

from zarr_tools.ngff.ngff_utils import (
    create_ome_metadata,
    get_spatial_dataset_voxel_spacing,
)
from zarr_tools.io.zarr_io import create_zarr_array

from .cli import floattuple
from .io_utils.read_utils import read_array_attrs
from .utils.configure_logging import configure_logging


logger:logging.Logger


def _define_args():
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument('--spots-source-image',
                             dest='spots_source_image',
                             type=str,
                             help='Path to the image container from which the spots were extracted')
    args_parser.add_argument('--spots-source-subpath', '--spots_source_subpath',
                             dest='spots_source_subpath',
                             type=str,
                             default=None,
                             help='Optional dataset subpath for the spots')
    args_parser.add_argument('--voxel-spacing', '--voxel_spacing',
                             dest='voxel_spacing',
                             type=floattuple,
                             help='Spots image voxel spacing')

    args_parser.add_argument('--expansion',
                             dest='expansion',
                             type=float,
                             default=1.0,
                             help='Spots volume expansion')

    args_parser.add_argument('--spots-file', '--spots_file',
                             dest='spots_file',
                             type=str,
                             required=True,
                             help='Spots image voxel spacing')

    args_parser.add_argument('--spots-output-image',
                             dest='spots_output_image',
                             type=str,
                             required=True,
                             help='Path to the output spots image file')

    args_parser.add_argument('--spots-image-subpath-reference',
                             dest='spots_image_subpath_reference',
                             type=str,
                             default=None,
                             help='Dataset subpath used for getting the shape of the output spots image. If no value is provided will use the same shape as the input')

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
    logger.info(f'Generate spots image with: {args}')

    spots_source_image_attrs = read_array_attrs(args.spots_source_image, args.spots_source_subpath)
    logger.info(f'Input image {args.spots_source_image}:{args.spots_source_subpath} attributes {spots_source_image_attrs}')
    spots_source_image_shape = spots_source_image_attrs['array_shape']

    if args.voxel_spacing is not None:
        # voxel spacing is specified in the command line, so use this value
        voxel_spacing = np.array(args.voxel_spacing[::-1]) # this is specified as XYZ and we want it as ZYX
        logger.info(f'Voxel spacing argument: {voxel_spacing}')
    else:
        voxel_spacing = get_spatial_dataset_voxel_spacing(spots_source_image_attrs, args.spots_source_subpath)
        logger.info(f'Voxel spacing for dataset {args.spots_source_subpath}: {voxel_spacing}')

    spots_dataset_reference = (args.spots_image_subpath_reference
                               if args.spots_image_subpath_reference
                               else args.spots_source_image_subpath)
    spots_reference_image_attrs = read_array_attrs(args.spots_source_image, spots_dataset_reference)
    logger.info(f'Ref image {args.spots_source_image}:{spots_dataset_reference} attributes {spots_reference_image_attrs}')

    spots_image_shape = spots_reference_image_attrs['array_shape']
    spots_reference_voxel_spacing = get_spatial_dataset_voxel_spacing(spots_reference_image_attrs, spots_dataset_reference)
    logger.info(f'Reference dataset {spots_dataset_reference} spacing: {spots_reference_voxel_spacing}')

    spots_xyz = _read_spots(args.spots_file)
    logger.info((
        f'Resample spots at {args.spots_source_subpath} from {spots_source_image_shape} image with spacing {voxel_spacing} '
        f'to {spots_dataset_reference} with shape {spots_image_shape} and spacing {spots_reference_voxel_spacing}'))
    spots_zyx = spots_xyz[:, :3][:, ::-1] / spots_reference_voxel_spacing * args.expansion

    output_spots_image = Path(args.spots_output_image)

    _generate_spots_image(
        spots_zyx,
        spots_image_shape,
        output_spots_image,
        spots_dataset_reference=spots_dataset_reference,
        reference_image_attrs=spots_reference_image_attrs,
    )


def _read_spots(spots_file):
    with open(spots_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()

    skiprows = 0
    if first_line:
        try:
            np.asarray(first_line.split(','), dtype=float)
        except ValueError:
            skiprows = 1

    return np.loadtxt(spots_file, delimiter=',', skiprows=skiprows, ndmin=2)


def _generate_spots_image(voxel_spots_zyx:np.ndarray,
                          image_shape:tuple,
                          output_path: Path,
                          spots_dataset_reference:str=None,
                          reference_image_attrs:dict=None):
    spatial_shape = image_shape[-3:]
    spots_image = np.zeros(spatial_shape, dtype=np.uint16)

    coords = voxel_spots_zyx[:, :3].astype(int)

    for coord in coords:
        spots_image[coord[0]-1:coord[0]+1,
                    coord[1]-1:coord[1]+1,
                    coord[2]-1:coord[2]+1] += 1

    logger.info(f'Writing spots to {spatial_shape} image to {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.zarr':
        _write_spots_as_ome_zarr(
            spots_image,
            output_path,
            spots_dataset_reference,
            reference_image_attrs,
        )
    else:
        import nrrd
        nrrd.write(str(output_path), spots_image.transpose(2,1,0), compression_level=2)


def _write_spots_as_ome_zarr(spots_image:np.ndarray,
                             output_path:Path,
                             dataset_subpath:str,
                             reference_image_attrs:dict):
    image_transforms = reference_image_attrs.get('array_transforms', {})
    scale = image_transforms.get('scale')
    translation = image_transforms.get('translation')

    ome_metadata = create_ome_metadata(
        os.path.basename(str(output_path)),
        dataset_subpath,
        reference_image_attrs.get('array_axes'),
        scale if scale else [1.0] * spots_image.ndim,
        translation if translation else [0.0] * spots_image.ndim,
        spots_image.ndim,
        ome_version='0.4',
    )

    logger.info(f'Writing OME-ZARR spots image to {output_path}:{dataset_subpath} with metadata {ome_metadata}')

    zarray = create_zarr_array(
        str(output_path),
        dataset_subpath,
        spots_image.shape,
        spots_image.shape,  # single chunk
        spots_image.dtype.name,
        overwrite=True,
        parent_array_attrs=ome_metadata,
    )
    zarray[:] = spots_image


if __name__ == '__main__':
    _main()
