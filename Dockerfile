# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM $BASE_CONTAINER
# Maintainer of the base container: LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

USER $NB_UID
COPY requirements.txt /tmp/
RUN pip install --default-timeout=15000 --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER


COPY mock_data mock_data
COPY flight_planner flight_planner
COPY app.ipynb .
