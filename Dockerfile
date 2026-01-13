FROM ghcr.io/janeliascicomp/dask:2025.11.0-py12-ol9
ARG SPOTS_TOOLS_BRANCH=main
ARG SPOTS_TOOLS_COMMIT=05b3785

WORKDIR /opt/scripts/spots-utils

ENV MKL_NUM_THREADS=
ENV NUM_MKL_THREADS=
ENV OPENBLAS_NUM_THREADS=
ENV OPENMP_NUM_THREADS=
ENV OMP_NUM_THREADS=

ENV PIP_ROOT_USER_ACTION=ignore

# Use the base environment from the baseImage and the conda-env
# from current dir
COPY conda-env.yaml .
RUN mamba env update -n base -f conda-env.yaml
RUN echo ${SPOTS_TOOLS_COMMIT} > .commit

# install bigstream
COPY configs configs
COPY src src

COPY *.py .
COPY *.toml .
COPY *.md .

RUN pip install --root-user-action ignore -e .
