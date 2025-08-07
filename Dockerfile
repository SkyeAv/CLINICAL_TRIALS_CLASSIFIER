# author --> Skye Lane Goetz

# mamba
FROM mambaorg/micromamba:1.6.2

WORKDIR /CLINICAL_TRIALS_CLASSIFIER
EXPOSE 8090

COPY . .

RUN micromamba env create -f /tmp/environment.yaml -n ct-classifier && micromamba clean --all --yes
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# typer cli
ENTRYPOINT ["cli"]
CMD ["--help"]