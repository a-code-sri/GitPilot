# Use Python 3.9 as base
FROM python:3.9-slim

# Set environment to non-interactive to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    gnupg \
    ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python Backend Dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install Frontend Dependencies
COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm install

# Build Frontend
COPY frontend/ ./
RUN npm run build

# Copy backend code AND app.py (which is in root)
WORKDIR /app
COPY backend/ ./backend/
COPY app.py ./

# Create repos directory
RUN mkdir -p /app/repos

# Expose ports
EXPOSE 5000 5173

# Create startup script (CHANGED: app.py is now in root, not backend/)
RUN echo '#!/bin/bash\n\
python app.py &\n\
cd /app/frontend && npm run dev -- --host 0.0.0.0\n\
' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]