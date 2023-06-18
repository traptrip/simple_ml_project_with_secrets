python preprocess.py && \
    tar -czvf data/humpback-whale-identification.tar.gz data/humpback-whale-identification && \
    dvc add data/humpback-whale-identification.tar.gz && \
    dvc push
