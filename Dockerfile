ARG GIT_HASH=fdcdc17

FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9 AS builder

# git is needed for the git-pinned PyPI deps (fishspot, zarr-tools)
RUN dnf install -y git

ARG PIXI_VERSION=v0.84.1
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=${PIXI_VERSION} PIXI_HOME=/opt/pixi bash
ENV PATH=/opt/pixi/bin:${PATH}

WORKDIR /opt/scripts/spots-utils
# install the locked pixi environment
COPY pyproject.toml pixi.lock ./
COPY src src
RUN pixi install --locked

FROM ghcr.io/janeliascicomp/dask:2026.6.0-py12-ol9
ARG GIT_HASH

WORKDIR /opt/scripts/spots-utils

ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

# Same path as in the builder: conda envs hardcode their prefix, and the
# editable install points at /opt/scripts/spots-utils/src.
# The pixi env goes first on PATH so python and dask resolve to it
# rather than to the base image's conda env.
COPY --from=builder /opt/scripts/spots-utils/.pixi/envs/default /opt/scripts/spots-utils/.pixi/envs/default
ENV PATH=/opt/scripts/spots-utils/.pixi/envs/default/bin:${PATH}

COPY configs configs
COPY src src

COPY *.toml .
COPY *.md .

RUN python -c "import SimpleITK as sitk; print('SITK with Elastix support' if hasattr(sitk, 'ElastixImageFilter') else 'SITK without Elastix support')"

RUN echo "${GIT_HASH}" > .commit
