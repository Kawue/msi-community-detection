FROM continuumio/anaconda3
COPY environment.yml .
COPY main.py .
COPY /kode ./kode
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/grine/bin:$PATH
RUN /bin/bash -c "source activate grine"
ENTRYPOINT ["python", "main.py"]