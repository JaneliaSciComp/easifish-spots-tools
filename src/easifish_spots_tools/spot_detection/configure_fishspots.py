import pydantic.v1.utils as pu
import yaml


def get_fishspots_config(config_filename):
    default_config = yaml.safe_load(_default_fishspots_config)
    if config_filename:
        with open(config_filename) as f:
            external_config = yaml.safe_load(f)
            print(f'Read external config from {config_filename}: {external_config}')
            config = pu.deep_update(default_config, external_config)
            print(f'Final config {config}')
    else:
        config = default_config

    return config


_default_fishspots_config="""
white_tophat_args:
  radius: 4

psf_estimation_args:
  n_propose_model: 10
  max_iterations: 1000
  inlier_threshold: 0.9
  min_inliers: 25
  radius: 9

deconvolution_args:
  clip: False
  num_iter: 20
  filter_epsilon: 0.000001

spot_detection_args:
  min_radius: 1
  max_radius: 6
  min_sigma:
  max_sigma:
  num_sigma: 5
  threshold:
  threshold_rel: 0.1
"""