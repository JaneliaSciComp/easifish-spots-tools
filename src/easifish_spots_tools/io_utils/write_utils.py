import logging
import os
import numpy as np

from pathlib import Path

from zarr_tools.ngff.ngff_utils import create_ome_metadata
from zarr_tools.io.zarr_io import create_zarr_array


logger = logging.getLogger(__name__)


def write_spots_image(voxel_spots_zyx:np.ndarray,
                      image_shape:tuple,
                      output_path: Path,
                      spots_dataset_reference:str=None,
                      reference_image_attrs:dict=None):
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
