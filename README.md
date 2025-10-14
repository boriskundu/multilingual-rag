# Multilingual RAG: Comparative Study for Low-Resource Languages

A research project comparing **Multilingual Embeddings** vs. **Translation Pipeline** approaches for Retrieval-Augmented Generation (RAG) in low-resource medical question-answering systems.

## 🎯 Overview

This project investigates two paradigms for building RAG systems for Hindi medical question-answering:

1. **Multilingual Embeddings Approach**: Direct processing in Hindi using multilingual embedding models
2. **Translation Pipeline Approach**: Translate query to English → retrieve → generate → translate back to Hindi

## 🔬 Research Context

Developed for submission to the **LLMs4All Workshop** at IEEE BigData 2025 Conference (Macau, December 8-11, 2025).

### Key Research Questions
- How do multilingual embeddings compare to translation pipelines for low-resource medical QA?
- What are the trade-offs between response quality, retrieval accuracy, and time efficiency?
- Can these approaches complement each other in a hybrid system?

## 📊 Key Findings

| Metric | Multilingual | Translation | Winner |
|--------|-------------|-------------|--------|
| **Avg Response Time** | 5.01s ± 1.51s | 2.87s ± 0.63s | Translation (74% faster)* |
| **Retrieval Quality** | 10.44 ± 2.21 | 16.54 ± 2.76 | Translation (58% better)* |
| **Semantic Similarity** | 0.611 ± 0.171 | - | - |
| **ROUGE-L F1** | 0.196 | 0.196 | Tied |
| **Retrieval Overlap** | 20% | 20% | Low overlap suggests complementary approaches |

*Statistically significant (p < 0.001)

## 🏗️ Project Structure

```
multilingual-rag/
├── data/                      # Medical corpora and datasets (excluded from git)
│   └── embeddings
|   └── processed
|   └── raw
│
├── logs/                      # Execution logs (excluded from git)
│   └── [data_collection_*.log]
│
├── mul-rag/                   # Virtual environment  (excluded from git)
│   └── [Libraries]
│
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── .ipynb_checkpoints/
│   ├── 1_data_collection.ipynb                # Data Collection
│   ├── 2_data_processing.ipynb                # Data Processing & Vectorization
│   ├── 3_multilingual_rag_implementation.ipynb                # Multilingual RAG Implementation. # SET THE API KEY IN ENVIRONMENT OPEN_API_KEY = "<Your key>"
│   └── 4_evaluation_and_analysis.ipynb                # Comparative Evaluation & Analysis
│
├── results/                   # Evaluation results and figures (excluded from git)
│   ├── figures/
│   │   ├── per_question.png           # Per-query performance breakdown
│   │   ├── quality_metrics.png        # Response quality distributions
│   │   ├── statistical_report.csv     # Statistical significance tests
│   │   └── time_comparison.png        # Latency analysis
│   ├── evaluation_*.csv               # Timestamped evaluation runs
│   └── hindi_healthcare_rag_*.csv     # Hindi medical QA results
│
├── src/                       # Source code modules
│   ├── __pycache__/
│   ├── .ipynb_checkpoints/
│   ├── __init__.py            # Package initialization
│   ├── collectors.py          # Data collection utilities
│   ├── data_processor.py      # Document preprocessing pipeline
│   ├── rag_system.py          # Core RAG implementation
│   └── utils.py               # Helper functions and utilities
│
├── .env                       # Environment variables (excluded from git)
├── .gitignore                 # Git exclusion rules
├── README.md                  # This file
└── requirements.txt           # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- OpenAI API key (for embeddings and LLM)
- Translation API key (optional, for translation pipeline)

### Installation

```bash
# Clone the repository
git clone https://github.com/boriskundu/multilingual-rag.git
cd multilingual-rag

# Create virtual environment
python -m venv mul-rag
source mul-rag/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install kernel with environment name
python -m ipykernel install --user --name mul-rag --display-name "MUL-RAG Research"

# Start Jupyter Lab
(mul-rag) C:\Users\Boris\Desktop\code\multilingual-rag>jupyter lab

# SET THE API KEY IN ENVIRONMENT in 3_multilingual_rag_implementation.ipynb 
OPEN_API_KEY = "<Your key>"

# Run the notebooks in order after selecting kernel "MUL-RAG Research" from list
1_data_collection.ipynb               
2_data_processing.ipynb                
3_multilingual_rag_implementation.ipynb                
4_evaluation_and_analysis.ipynb 
```

### Environment Setup (Optional)

Create a `.env` file in the project root:

```env
# OpenAI API (for embeddings/generation)
OPENAI_API_KEY=your_key_here
```

### Running Experiments

Execute notebooks in sequential order:

#### **Notebook 1: Data Collection** (`1_data_collection.ipynb`)
- Collect medical documents from various sources
- Download FDA drug labels and medical corpora
- Validate data quality and format
- **Output**: Raw medical documents in `data/`

#### **Notebook 2: Data Processing & Vectorization** (`2_data_processing.ipynb`)
- Clean and preprocess medical text
- Chunk documents for optimal retrieval
- Generate embeddings (multilingual and English)
- Build vector stores
- **Output**: Processed medical documents and embeddings in `data/`

#### **Notebook 3: Multilingual RAG Implementation** (`3_multilingual_rag_implementation.ipynb`)
- Implement multilingual embeddings approach
- Implement translation pipeline approach
- Test both systems on Hindi medical queries
- **Output**: System responses in `results/`

#### **Notebook 4: Comparative Evaluation & Analysis** (`4_evaluation_and_analysis.ipynb`)
- Compare approaches across all metrics
- Statistical significance testing
- Generate visualizations and reports
- Qualitative error analysis
- **Output**: All figures and CSV files in `results/`

## 📈 Evaluation Metrics

### Response Quality
- **Semantic Similarity**: Cosine similarity in multilingual embedding space (no translation bias)
- **ROUGE Scores**: Lexical overlap on English translations (ROUGE-1, ROUGE-2, ROUGE-L)

### Efficiency
- **Response Time**: End-to-end latency for query processing
- **Time Breakdown**: Separate timing for retrieval, generation, and translation steps

### Retrieval Quality
- **Average Chunk Score**: Relevance scores of retrieved document chunks
- **Retrieval Overlap**: Percentage of shared retrieved chunks between approaches

## 🔍 Key Insights

### Translation Pipeline Strengths
- ✅ **Significantly faster** (74% speed improvement, p=0.0006)
- ✅ **Better retrieval quality** (58% higher relevance scores, p<0.0001)
- ✅ **More consistent latency** (lower variance)
- ✅ **Leverages abundant English medical data**

### Multilingual Embeddings Strengths
- ✅ **No translation overhead** in production
- ✅ **Preserves linguistic nuances** (potentially)
- ✅ **Direct native language processing**
- ✅ **No dependency on translation APIs**

### Critical Finding: Low Retrieval Overlap (20%)
The two approaches retrieve fundamentally different information, suggesting they may be **complementary rather than competing**. This opens possibilities for hybrid ensemble approaches.

## 🎯 Conference Alignment

This research directly addresses LLMs4All Workshop themes:

- **Cross-Lingual and Multilingual Learning**: Comparative study of knowledge transfer approaches
- **Efficient and Inclusive Model Training**: Resource-efficient strategies for low-resource languages
- **Real-World Applications**: Medical question-answering in Hindi
- **Data Scarcity Solutions**: Leveraging high-resource language data for low-resource applications
- **Retrieval-Augmented Generation**: Novel comparison of RAG paradigms
- **Benchmarking and Evaluation**: Creation of evaluation framework for multilingual RAG systems

## 📊 Results Visualization

The `results/figures/` directory contains:

1. **time_comparison.png**: Breakdown of latency by pipeline stage
2. **quality_metrics.png**: Distribution of semantic similarity and ROUGE scores
3. **per_question.png**: Question-by-question performance comparison
4. **statistical_report.csv**: Formal statistical analysis with p-values

## 🛣️ Future Work

### Immediate Next Steps
1. **Medical Accuracy Validation**: Expert annotation of response correctness
2. **Qualitative Error Analysis**: Deep dive into the 20% retrieval overlap
3. **Multi-language Extension**: Test on Cantonese, Vietnamese, Tagalog
4. **Linguistic Quality Assessment**: Native speaker evaluation of response naturalness

### Long-term Directions
1. **Hybrid Ensemble Approach**: Combine both methods for optimal performance
2. **Benchmark Creation**: Release Hindi medical QA evaluation dataset
3. **Scalability Study**: Test with 10K+ documents
4. **Real-world Deployment**: User study with Hindi-speaking patients
5. **Cost-Benefit Analysis**: Production deployment economics

## 🔧 Source Code Modules

### `src/collectors.py`
Data collection utilities for scraping and downloading medical documents from various sources.

### `src/data_processor.py`
Document preprocessing pipeline including:
- Text cleaning and normalization
- Document chunking strategies
- Metadata extraction
- Quality filtering

### `src/rag_system.py`
Core RAG implementation with:
- Multilingual embedding approach
- Translation pipeline approach
- Retrieval mechanisms
- Response generation
- Evaluation harness

### `src/utils.py`
Helper functions for:
- API interactions
- File I/O operations
- Logging and debugging
- Performance measurement

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{yourname2025multilingual,
  title={Bridging the Medical Information Gap: Comparing Translation-Based and Multilingual RAG Approaches for Low-Resource Languages},
  author={Boris Kundu},
  booktitle={LLMs4All Workshop at IEEE BigData 2025},
  year={2025},
  address={Macau}
}
```

## 📄 License

[Your chosen license - e.g., MIT, Apache 2.0]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines
1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Include tests for new features
4. Update documentation as needed

## ⚠️ Limitations

- Limited to Hindi language (expansion to other languages planned)
- Medical domain only (generalization to other domains unexplored)
- Depends on proprietary APIs (OpenAI) for core functionality
- Small evaluation set (12 questions)
- No human expert validation of medical accuracy yet

## 📧 Contact

[Boris Kundu]  
[boriskundu@gmail.com]  
[Your institution/affiliation]

## 🙏 Acknowledgments

- LLMs4All Workshop organizers
- IEEE BigData 2025 Conference
- [Any funding sources or collaborators]
- OpenAI for API access
- Medical document sources (FDA, etc.)

---

**Note**: This is an active research project. Results are preliminary and subject to peer review. The system is designed for research purposes only and should not be used for actual medical advice.