FROM sinpcw/pytorch:1.8.0

USER root

RUN pip install -U pip && \
    pip install streamlit && \
    pip install effdet && \
    pip install git+https://github.com/alexhock/object-detection-metrics

WORKDIR ./