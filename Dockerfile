# Use an official Python slim runtime as the parent image
FROM python:3.11-slim AS builder

# Set shell and environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build dependencies if needed (none are strictly required for pure python packages here)
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt


# Final stage
FROM python:3.11-slim AS runner

WORKDIR /app

# Create a non-root user for security
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

# Copy installed pip packages from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appgroup . .

# Set paths and user context
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Ensure outputs directory exists and is owned by the appuser
RUN mkdir -p /app/outputs && chown -R appuser:appgroup /app/outputs

USER appuser

# Expose Streamlit port
EXPOSE 8501

# Add standard healthcheck for the web app
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Start the Streamlit application by default
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]
