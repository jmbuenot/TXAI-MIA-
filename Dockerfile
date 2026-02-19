FROM jupyter/datascience-notebook:python-3.11

USER root

RUN apt-get update

RUN apt-get update \
    && echo "Updated apt-get" \
    && apt-get install -y openjdk-8-jre \
    && echo "Installed openjdk 8"

USER ${NB_UID}

RUN pip install pyppeteer
RUN pyppeteer-install

RUN conda install -y -q -c conda-forge pydot python-graphviz
RUN conda install -y -q -c conda-forge py4j

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install xgboost dice_ml --quiet

WORKDIR "${HOME}"