#!/bin/bash
cat <<EOF > /tmp/myray-image.dockerfile
FROM rayproject/ray:2.44.1-py311
RUN pip install numpy && pip cache purge
RUN pip install -U cupy-cuda12x scipy numpy>=2.0 && pip cache purge
ENV PYTHON_VERSION=3.11
EOF
docker build -f /tmp/myray-image.dockerfile -t registry.cn-hangzhou.aliyuncs.com/lacogito/ray:2.44.1-py311-numpy2x .
