# ===========================================
# Stage 1: Base system setup
# ===========================================
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Prevent interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# ===========================================
# Stage 2: App setup
# ===========================================
WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install dependencies (cached if requirements.txt unchanged)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your project
COPY . .

# Expose Gradio’s default port
EXPOSE 7860

# (Optional) environment variables can also be passed at runtime
# ENV OPENAI_API_KEY=your_key
# ENV HF_TOKEN=your_token

# Default command
CMD ["python3", "-m", "src.app.app"]
