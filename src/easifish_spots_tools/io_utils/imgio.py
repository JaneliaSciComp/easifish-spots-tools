import os
import zarr

from tifffile import TiffFile
from zarr_tools.io.zarr_io import (open_zarr_store, read_zarr_block)


def open_image_array(container_path, subpath,
                     data_timeindex=None, data_channels=None):
    real_container_path = os.path.realpath(container_path)
    path_comps = os.path.splitext(container_path)

    container_ext = path_comps[1]

    if container_ext == '.tif' or container_ext == '.tiff':
        print(f'Open tiff {container_path} ({real_container_path})')
        tiff_attrs = _read_tiff_attrs(real_container_path)
        return _open_tiff_array(real_container_path), tiff_attrs
    elif (container_ext == '.n5'
          or os.path.exists(f'{real_container_path}/attributes.json')):
        print((
            f'Open N5 {container_path} ({real_container_path}) '
            f'subpath: {subpath} '
            f'timeindex: {data_timeindex} '
            f'channels: {data_channels} '
        ))
        return _open_zarr(real_container_path, subpath,
                          data_timeindex=data_timeindex,
                          data_channels=data_channels)
    elif container_ext == '.zarr':
        print((
            f'Open Zarr {container_path} ({real_container_path}) '
            f'subpath: {subpath} '
            f'timeindex: {data_timeindex} '
            f'channels: {data_channels} '
        ))
        return _open_zarr(real_container_path, subpath,
                          data_timeindex=data_timeindex,
                          data_channels=data_channels)
    else:
        print(f'Cannot handle {container_path} ({real_container_path}) {subpath}')
        return None, {}


def _open_tiff_array(input_path):
    tif = TiffFile(input_path)
    tif_store = tif.aszarr()
    return zarr.open(tif_store)


def _read_tiff_attrs(input_path):
    with TiffFile(input_path) as tif:
        tif_store = tif.aszarr()
        tif_array = zarr.open(tif_store)
        return {
            'array_axes': tif.series[0].axes,
            'array_storepath': input_path,
            'array_subpath': '',
            'array_shape': tif_array.shape,
            'array_ndim': tif_array.ndim,
            'array_dtype': tif_array.dtype.name,
            'array_chunksize': tif_array.chunks,
        }


def _open_zarr(data_path, data_subpath,
               data_timeindex=None, data_channels=None, 
               mode='r',
               block_coords=None):
    try:
        print(f'Opening {data_path}:{data_subpath}:{data_timeindex}:{data_channels} at {block_coords}')
        zstore, zattrs = open_zarr_store(data_path, data_subpath)
        zarray_subpath = zattrs['array_subpath']
        zarray = zarr.open_array(store=zstore, path=zarray_subpath, mode=mode)
        return read_zarr_block(zarray, zattrs, data_timeindex, data_channels, block_coords), zattrs
    except Exception as e:
        print(f'Error reading {data_path}:{data_subpath}:{data_timeindex}:{data_channels}:{block_coords}', e)
        raise e
