FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt-get update \
  && apt-get install -yqq --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/sshd

RUN pip install mpi4py netcdf4 xarray h5py ruamel.yaml einops wandb segmentation-models-pytorch imageio[ffmpeg] dask_labextension jupyter-server-proxy jupyterlab_code_formatter nbresuse && pip install --upgrade jupyterlab jupytext imageio opencv-python
