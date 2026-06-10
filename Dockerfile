FROM python:3.12-slim

WORKDIR /code

# Entry scripts live in subfolders (grabers/...) but import top-level modules
# (helpers, constants) — make /code importable regardless of script location.
ENV PYTHONPATH=/code

# Install Python dependencies (rebuild only when requirements.txt changes).
# TA-Lib comes from the PyPI wheel (>=0.6 bundles the C library) — no source build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code is mounted at /code via volume — NOT copied into image
# No CMD — every service defines its own command:
