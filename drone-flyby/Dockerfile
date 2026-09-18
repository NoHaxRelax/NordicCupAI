FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-detector.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The detector stack (torch, ultralytics) is large and GPU-specific. Build with
# --build-arg WITH_DETECTOR=1 for a CPU image, or install it in the host
# environment and run api.py directly on a GPU machine.
ARG WITH_DETECTOR=0
RUN if [ "$WITH_DETECTOR" = "1" ]; then pip install --no-cache-dir -r requirements-detector.txt; fi

# The server, the tracker runtime and the detector adapters. The sample data
# under src/ is not copied: it is 400 MB and only local_evaluator.py needs it.
COPY dtos.py utils.py example.py api.py detectors.py ./
COPY tracking/ tracking/

# DRONE_WEIGHTS must point at a checkpoint mounted into the container.
EXPOSE 9053

CMD ["python", "api.py"]
