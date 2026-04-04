# Customer Sentiment Analysis - MLOps Project

Professional MLOps structure for customer sentiment analysis using machine learning and deep learning techniques.

## 📁 Project Structure

```
CustomerSentimentAnalysis/
├── .github/
│   └── workflows/              # GitHub Actions CI/CD workflows
├── config/                      # Configuration files
├── data/
│   ├── raw/                    # Raw data (not tracked by Git)
│   └── processed/              # Processed data
├── src/                         # Source code
│   ├── api/                    # FastAPI application
│   ├── data/                   # Data loading and preprocessing
│   ├── features/               # Feature engineering
│   └── models/                 # Model training and evaluation
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Unit and integration tests
├── models/                      # Trained models (not tracked by Git)
├── mlruns/                      # MLflow experiment tracking
├── Dockerfile                   # Docker containerization
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata and build config
├── Makefile                    # Common development tasks
└── dvc.yaml                    # DVC pipeline configuration
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
make install
```

### 2. Prepare Data
```bash
make dvc-pipeline
```

### 3. Train Model
```bash
make train
```

### 4. Run Tests
```bash
make test
```

### 5. Start API Server
```bash
make serve
```

## 📊 Available Commands

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make train` | Train the model |
| `make test` | Run test suite with coverage |
| `make lint` | Check code quality |
| `make format` | Format code with black and isort |
| `make clean` | Remove build artifacts |
| `make serve` | Start FastAPI development server |
| `make dvc-pipeline` | Run data pipeline |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker container |

## 🐳 Docker Deployment

Build and run the application in Docker:

```bash
# Build image
make docker-build

# Run container
make docker-run
```

The API will be available at `http://localhost:8000`

## 📈 MLOps Tools Integration

- **MLflow**: Experiment tracking and model registry (`mlruns/`)
- **DVC**: Data versioning and pipeline automation (`dvc.yaml`)
- **Docker**: Containerization for reproducible deployments
- **GitHub Actions**: CI/CD automation

## 📝 Development Workflow

1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and commit: `git commit -am "Add feature"`
3. Run tests locally: `make test`
4. Push and create pull request: `git push origin feature/new-feature`
5. GitHub Actions automatically runs CI/CD pipeline

## 🔧 Configuration

- **Model config**: `config/model.yaml`
- **Data config**: `config/data.yaml`
- **API config**: Environment variables (`.env`)

## 📚 Core Modules

### `src/data/`
- Data loading and initial preprocessing
- Data validation and quality checks

### `src/features/`
- Feature engineering and transformation
- Feature selection and importance analysis

### `src/models/`
- Model architecture definitions
- Training loops and evaluation metrics

### `src/api/`
- FastAPI endpoints
- Model serving and inference

## 🧪 Testing

Run tests with coverage:
```bash
pytest tests/ -v --cov=src
```

## 🔍 Code Quality

Format code:
```bash
make format
```

Lint code:
```bash
make lint
```

## 📦 Dependencies

- **Data Processing**: pandas, numpy, scikit-learn
- **NLP**: transformers, nltk, torch
- **API**: fastapi, uvicorn, pydantic
- **MLOps**: mlflow, dvc, wandb
- **Development**: pytest, black, flake8, isort

## 📄 License

MIT

## 👥 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Last Updated**: 2026-04-04
