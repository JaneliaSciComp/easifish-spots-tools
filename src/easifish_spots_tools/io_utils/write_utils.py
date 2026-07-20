import logging
import os
import numpy as np

from pathlib import Path

from typing import Optional

from zarr_tools.ngff.ngff_utils import create_ome_metadata, get_axes
from zarr_tools.io.zarr_io import create_zarr_array


logger = logging.getLogger(__name__)


def write_spots_image(voxel_spots_zyx:np.ndarray,
                      image_shape:tuple,
                      output_path: Path,
                      spots_dataset_reference:Optional[str]=None,
                      reference_image_attrs:dict={}):
    """Create a spots density image from voxel coordinates and write it to disk.

    Supports .zarr (with OME metadata) and other formats via nrrd.
    """
    spatial_shape = image_shape[-3:]
    spots_image = np.zeros(spatial_shape, dtype=np.uint16)

    coords = voxel_spots_zyx[:, :3].astype(int)

    for coord in coords:
        spots_image[coord[0]-1:coord[0]+1,
                    coord[1]-1:coord[1]+1,
                    coord[2]-1:coord[2]+1] += 1

    logger.info(f'Writing spots to {spatial_shape} image to {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.zarr2' or output_path.suffix == '.zarr' or output_path.suffix == '.zarr3':
        zarr_format = 2 if output_path.suffix == '.zarr2' else 3
        _write_spots_as_ome_zarr(
            spots_image,
            output_path,
            spots_dataset_reference,
            reference_image_attrs,
            zarr_format=zarr_format
        )
    else:
        import nrrd
        nrrd.write(str(output_path), spots_image.transpose(2,1,0), compression_level=2)


def _write_spots_as_ome_zarr(spots_image:np.ndarray,
                             output_path:Path,
                             dataset_subpath:Optional[str],
                             reference_image_attrs:dict,
                             zarr_format:int=2):
    logger.debug(f'Extract OME axes and transforms from {reference_image_attrs}')
    axes = get_axes(reference_image_attrs)
    image_transforms = reference_image_attrs.get('array_transforms', {})
    scale = image_transforms.get('scale')
    translation = image_transforms.get('translation')

    ome_metadata = create_ome_metadata(
        os.path.basename(str(output_path)),
        dataset_subpath,
        axes,
        scale if scale[-len(axes):] else [1.0] * spots_image.ndim,
        translation if translation[-len(axes):] else [0.0] * spots_image.ndim,
        spots_image.ndim,
        ome_version='0.4' if zarr_format == 2 else '0.5',
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
        zarr_format=zarr_format,
    )
    zarray[:] = spots_image
