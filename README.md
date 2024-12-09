# EffiScale: Orchestrated Elasticity Framework for Cloud-native Container Scaling

<div align="center">

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](https://www.docker.com/)

</div>

EffiScale is a self-adaptive framework designed to optimize both vertical and horizontal container scaling within cloud infrastructure. It employs a decentralized microservice architecture based on IBM's MAPE-K reference model to provide intelligent and efficient container elasticity management.

## Key Features

- **Decentralized Microservice Architecture**
  - Independent scaling components 
  - Fault-tolerant design
  - Eliminates single points of failure
  - Enables localized decision-making

- **Hybrid Scaling Capabilities**
  - Seamless vertical & horizontal scaling
  - Dynamic resource adjustment
  - Container instance management
  - Fine-grained elasticity control

- **Adaptive Thresholds**
  - Dynamic threshold adjustment
  - Real-time adaptation
  - Historical trend analysis
  - Precise scaling decisions

- **Enhanced Fault Tolerance**
  - Decentralized controllers
  - Automatic failover
  - Self-healing capabilities
  - High availability

## Architecture

The framework consists of four main components:

1. **Monitor Service** (`src/microservices/monitor_service.py`)
   - Resource utilization tracking
   - Metric collection
   - Performance monitoring
   - Real-time analytics

2. **Controller Service** (`src/microservices/controller_service.py`) 
   - Scaling decisions
   - Load balancing
   - Resource optimization
   - Federation management

3. **Decision Maker** (`src/controllers/decision_maker.py`)
   - Intelligent scaling logic
   - Predictive analytics
   - Pattern recognition
   - Adaptive learning

4. **Execution Service** (`src/microservices/execution_service.py`)
   - Scaling action execution
   - Container management
   - Resource allocation
   - State verification

## Quick Start

### Prerequisites

- Python 3.8+
- Docker
- FastAPI
- uvicorn

### Installation

```bash
# Clone the repository
git clone https://github.com/ahmadpanah/EffiScale.git

# Navigate to project directory 
cd EffiScale

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Start the EffiScale system
python main.py
```

This will start:
- Monitor Service on port 8001
- Controller Service on port 8002  
- Execution Service on port 8003

## Docker Deployment

### Using Docker Compose (Recommended)

1. Build and start services:
```bash
# Build the images
docker-compose build

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down

### API Endpoints

Monitor Service (Port 8001):
```bash
GET /metrics/{container_id}        # Get container metrics
POST /monitor/start/{container_id} # Start monitoring container
POST /monitor/stop/{container_id}  # Stop monitoring container
```

Controller Service (Port 8002):
```bash
POST /decision                     # Make scaling decision
GET /health                       # Check controller health
POST /register                    # Register new controller
```

Execution Service (Port 8003):
```bash
POST /scale/vertical              # Execute vertical scaling
POST /scale/horizontal           # Execute horizontal scaling
GET /history/{container_id}      # Get scaling history
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Seyed Hossein Ahmadpanah - h.ahmadpanah@iau.ac.ir