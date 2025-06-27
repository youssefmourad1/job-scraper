# Docker Deployment Guide

This guide explains how to deploy the Job Market Analyzer using Docker on any platform.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier deployment)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone the repository and navigate to the project directory:**
   ```bash
   cd /path/to/job-market-analyzer
   ```

2. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   Open your browser and go to `http://localhost:8501`

### Option 2: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t job-market-analyzer .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/cache:/app/cache \
     job-market-analyzer
   ```

3. **Access the application:**
   Open your browser and go to `http://localhost:8501`

## Platform-Specific Instructions

### Windows (PowerShell)
```powershell
# Build
docker build -t job-market-analyzer .

# Run with volume mounts
docker run -p 8501:8501 `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/logs:/app/logs `
  -v ${PWD}/cache:/app/cache `
  job-market-analyzer
```

### macOS/Linux
```bash
# Build
docker build -t job-market-analyzer .

# Run with volume mounts
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/cache:/app/cache \
  job-market-analyzer
```

## Environment Variables

You can customize the application behavior using environment variables:

```bash
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8501 \
  -e STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
  -e SELENIUM_HEADLESS=true \
  -e CHROME_BIN=/usr/bin/chromium \
  -v $(pwd)/data:/app/data \
  job-market-analyzer
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit server port |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Streamlit server address |
| `SELENIUM_HEADLESS` | `true` | Run browser in headless mode |
| `CHROME_BIN` | `/usr/bin/chromium` | Chrome/Chromium binary path |
| `CHROMEDRIVER_PATH` | `/usr/bin/chromedriver` | ChromeDriver binary path |
| `DISPLAY` | `:99` | X11 display for GUI applications |

## Production Deployment

### Using Docker Compose for Production

1. **Create a production docker-compose.yml:**
   ```yaml
   version: '3.8'
   services:
     job-market-analyzer:
       build: .
       ports:
         - "80:8501"
       volumes:
         - ./data:/app/data
         - ./logs:/app/logs
       restart: always
       environment:
         - STREAMLIT_SERVER_PORT=8501
         - STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

2. **Deploy:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Use a different port
   docker run -p 8502:8501 job-market-analyzer
   ```

2. **Permission issues with volumes:**
   ```bash
   # Create directories with proper permissions
   mkdir -p data logs cache
   chmod 755 data logs cache
   ```

3. **NLTK data not downloading:**
   The Dockerfile automatically downloads required NLTK data. If issues persist, rebuild the image:
   ```bash
   docker build --no-cache -t job-market-analyzer .
   ```

4. **Browser automation issues:**
   If Selenium/browser automation fails, check the browser setup:
   ```bash
   # Test browser setup inside container
   docker exec -it job-market-analyzer python -c "from utils.browser_setup import test_browser_setup; print(test_browser_setup())"
   
   # Check browser environment
   docker exec -it job-market-analyzer python -c "from utils.browser_setup import validate_browser_environment; print(validate_browser_environment())"
   ```

5. **Memory issues with browser automation:**
   Increase Docker memory limits:
   ```bash
   docker run --memory=2g --memory-swap=2g -p 8501:8501 job-market-analyzer
   ```

### Viewing Logs

```bash
# View container logs
docker logs job-market-analyzer

# Follow logs in real-time
docker logs -f job-market-analyzer
```

### Accessing the Container

```bash
# Execute bash in running container
docker exec -it job-market-analyzer bash

# Or if bash is not available
docker exec -it job-market-analyzer sh
```

## Health Checks

The container includes health checks. You can check the status:

```bash
# Check container health
docker ps

# View health check logs
docker inspect job-market-analyzer | grep -A 10 Health
```

## Updating the Application

1. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

2. **Rebuild and restart:**
   ```bash
   docker-compose down
   docker-compose up --build -d
   ```

## Resource Requirements

- **Minimum:** 1GB RAM, 1 CPU core
- **Recommended:** 2GB RAM, 2 CPU cores
- **Storage:** 500MB for application + data storage

## Security Considerations

- The application runs as a non-root user inside the container
- Only port 8501 is exposed
- No sensitive data is stored in the image
- Use environment variables for configuration in production

## Support

If you encounter issues:
1. Check the logs using `docker logs job-market-analyzer`
2. Ensure all required ports are available
3. Verify Docker and Docker Compose versions are up to date
4. Check system resources (RAM, disk space) 