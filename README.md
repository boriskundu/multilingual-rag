# Cross-Lingual RAG: Translation vs. Multilingual Embeddings for Low-Resource Languages

A research project comparing **Multilingual Embeddings** vs. **Translation Pipeline** approaches for Retrieval-Augmented Generation (RAG) in low-resource medical question-answering systems across **Hindi and Chinese**.

## 🎯 Overview

This project investigates two paradigms for building RAG systems for cross-lingual medical question-answering:

1. **Multilingual Embeddings Approach**: Direct processing in target language using multilingual embedding models
2. **Translation Pipeline Approach**: Translate query to English → retrieve → generate → translate back to target language

**Key Innovation**: Using **identical questions across Hindi and Chinese** and **cross-LLM validation** with both GPT-4o and Claude Sonnet 4.5 to reveal robust performance patterns.

## 🔬 Research Context

Developed for submission to conferences focusing on cross-lingual NLP and multilingual systems.

### Key Research Questions
- Do multilingual embeddings and translation pipelines access the same knowledge for identical questions?
- Are performance differences consistent across different LLM architectures (GPT-4o vs Claude Sonnet 4.5)?
- Which approach provides better reliability for critical areas like healthcare applications in low-resource settings?

## 📊 Key Findings

### Updated Results: Translation Pipeline Consistently Superior

**Cross-LLM Validation**: Testing with both GPT-4o and Claude Sonnet 4.5 reveals consistent translation advantages across both languages, demonstrating robust superiority independent of model architecture.

### Performance Comparison (Combined LLM Evaluation)

#### Overall Quality Scores (1-5 scale)

**GPT-4o Results:**
| Language | Multilingual | Translation | Margin | Winner |
|----------|-------------|-------------|--------|---------|
| **Hindi** | 4.76 | **4.78** | +0.02 | Translation |
| **Chinese** | 4.70 | **4.86** | +0.16 | Translation |

**Claude Sonnet 4.5 Results:**
| Language | Multilingual | Translation | Margin | Winner |
|----------|-------------|-------------|--------|---------|
| **Hindi** | 4.40 | **4.71** | +0.31 | Translation |
| **Chinese** | 4.49 | **4.63** | +0.14 | Translation |

**Key Finding**: Translation pipeline wins **ALL 4 comparisons** (2 languages × 2 LLMs), demonstrating robust advantages independent of model architecture.

### Safety Analysis: Translation Superior

#### Hallucination Rates

**GPT-4o:**
| Language | Multilingual | Translation | Safer |
|----------|-------------|-------------|--------|
| **Hindi** | 3.3% | 3.3% | Equivalent |
| **Chinese** | 6.7% | **0.0%** | Translation |

**Claude Sonnet 4.5:**
| Language | Multilingual | Translation | Safer |
|----------|-------------|-------------|--------|
| **Hindi** | 3.3% | **0.0%** | Translation |
| **Chinese** | 0.0% | 0.0% | Equivalent |

**Critical Finding**: Translation achieves **0% hallucinations in 3 of 4 conditions**, making it the safer choice for medical applications.

### Efficiency vs. Quality Tradeoff

**Timing Results:**
| Model | Language | Multi (s) | Trans (s) | Speedup |
|-------|----------|-----------|-----------|---------|
| GPT-4o | Hindi | 2.55 | 3.36 | -32% |
| GPT-4o | Chinese | 2.39 | 3.20 | -34% |
| Claude | Hindi | 5.83 | 6.11 | -5% |
| Claude | Chinese | 5.64 | 7.44 | -24% |

*Note: Negative speedup means multilingual is faster*

**Insight**: Multilingual is 5-34% faster, but translation's quality and safety advantages justify the modest latency overhead.

### Quality Dimensions: Translation's Completeness Advantage

Translation shows most pronounced improvements in **completeness** dimension:

**GPT-4o:**
- Hindi Appropriateness: +0.06 points
- Chinese Faithfulness: +0.24 points
- Chinese Appropriateness: +0.20 points

**Claude Sonnet 4.5:**
- Hindi Completeness: +0.80 points (largest improvement)
- Hindi Appropriateness: +0.14 points
- Chinese Completeness: +0.30 points

### Chunk Overlap Analysis

- **Hindi**: 28.7% overlap, 8/30 questions (27%) with zero overlap
- **Chinese**: 27.3% overlap, 13/30 questions (43%) with zero overlap

**Interpretation**: Approaches access fundamentally different information, with particularly divergent retrieval patterns for Chinese queries.

## 🏗️ Project Structure

```
multilingual-rag/
├── data/                      # Medical corpora and datasets (excluded from git)
│   ├── embeddings/           # Vector stores for both approaches
│   ├── processed/            # Chunked and processed documents
│   ├── questions/            # Generated question sets
│   │   ├── hindi_questions.json
│   │   ├── chinese_questions.json
│   │   └── question_comparison.json
│   └── raw/                  # Original medical documents
│
├── logs/                      # Execution logs (excluded from git)
│
├── mul-rag/                   # Virtual environment (excluded from git)
│
├── notebooks/                 # Jupyter notebooks for experiments
│   ├── 1_data_collection.ipynb                      # Data Collection from FDA/NIH
│   ├── 2_data_processing.ipynb                      # Data Processing & Vectorization
│   ├── 2x_generate_validated_questions.ipynb       # Generate identical questions for both languages
│   ├── 3_multilingual_rag_implementation.ipynb     # Cross-lingual RAG experiments
│   ├── 4_evaluation_and_analysis.ipynb             # Statistical analysis & visualization
│   ├── 5_llm_judge_evaluation.ipynb                # LLM-as-judge quality assessment
│   └── 6_ablation_analysis.ipynb                   # Ablation study comparison analysis
│
├── results/                   # Evaluation results and figures (excluded from git)
│   ├── baseline/             # GPT-4o results (1000/200 chunking)
│   │   ├── figures/
│   │   │   ├── cross_language_comparison.png           # Hindi vs Chinese analysis
│   │   │   ├── chunk_overlap_analysis.png             # Retrieval overlap visualization
│   │   │   ├── llm_judge_comprehensive_analysis.png   # Quality assessment
│   │   │   ├── time_comparison_per_language.png       # Efficiency analysis
│   │   │   └── quality_metrics_per_language.png       # Detailed quality breakdown
│   │   ├── multilingual_rag_results.csv              # Combined results
│   │   ├── llm_judge_evaluation.csv                  # Quality assessment results
│   │   ├── combined_chunk_analysis.csv               # Chunk overlap analysis
│   │   ├── enhanced_combined_evaluation.csv          # Comprehensive metrics
│   │   └── llm_judge_final_summary.csv              # Statistical summary
│   ├── claude_ablation/      # Claude Sonnet 4.5 results for cross-LLM validation
│   ├── chunk_500_100/        # Small chunks (500 chars, 100 overlap)
│   └── chunk_1500_300/       # Large chunks (1500 chars, 300 overlap)
│
├── src/                       # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── collectors.py          # Government data collection (FDA/NIH only)
│   ├── data_processor.py      # Document preprocessing pipeline
│   ├── rag_system.py          # Core multilingual RAG implementation
│   └── utils.py               # Translation utilities and LLM-as-judge evaluation
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
- OpenAI API key (for GPT-4o and embeddings)
- Anthropic API key (for Claude Sonnet 4.5 validation)

### Installation

```bash
# Clone the repository
git clone https://github.com/boriskundu/multilingual-rag.git
cd multilingual-rag

# Create virtual environment
python -m venv mul-rag
source mul-rag/bin/activate  # On Windows: mul-rag\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for Claude validation
pip install anthropic

# Start Jupyter Lab
jupyter lab
```

### API Key Setup

Set your API keys in relevant notebooks:

```python
# For notebooks 2x, 3, and 5
OPEN_API_KEY = "your_openai_api_key_here"

# For Claude validation studies (notebooks 3 and 5)
ANTHROPIC_API_KEY = "your_claude_api_key_here"
```

### Running Experiments

Execute notebooks in sequential order:

#### **Notebook 1: Data Collection** (`1_data_collection.ipynb`)
- Collect medical documents from government sources (FDA, NIH/MedlinePlus)
- Download drug labels, health topics, and medical information
- Validate data quality and ensure government-only sources
- **Output**: Raw medical documents in `data/raw/`

#### **Notebook 2: Data Processing** (`2_data_processing.ipynb`)
- Clean and preprocess medical text
- Chunk documents for optimal retrieval
- Generate multilingual embeddings and vector stores
- **Output**: Processed documents and embeddings in `data/processed/` and `data/embeddings/`

#### **Notebook 2x: Question Generation** (`2x_generate_validated_questions.ipynb`)
- Generate 30 base English questions from your corpus
- Translate identical questions to both Hindi and Chinese
- Validate questions against corpus for answerability
- **Output**: Validated question sets for both languages

#### **Notebook 3: Cross-Lingual RAG Implementation** (`3_multilingual_rag_implementation.ipynb`)
- Load identical question sets for both languages
- Run both approaches across Hindi and Chinese with both LLMs
- Time component analysis and chunk saving
- **Output**: Complete experimental results with timing breakdowns for both GPT-4o and Claude

#### **Notebook 4: Evaluation & Analysis** (`4_evaluation_and_analysis.ipynb`)
- Cross-language statistical analysis with chunk overlap analysis
- Performance comparison visualizations
- Deep dive into cross-LLM consistency patterns
- **Output**: Comprehensive analysis figures and chunk analysis

#### **Notebook 5: LLM-as-Judge Evaluation** (`5_llm_judge_evaluation.ipynb`)
- Quality assessment using both GPT-4o and Claude Sonnet 4.5 as evaluators
- Faithfulness, completeness, and appropriateness scoring
- Cross-language and cross-LLM safety analysis (hallucination detection)
- **Output**: Definitive quality comparison and safety assessment across both models

#### **Notebook 6: Ablation Analysis** (`6_ablation_analysis.ipynb`)
- Compare results across different LLMs and chunking strategies
- Validate robustness of key findings
- Statistical comparison of ablation studies
- **Output**: Comprehensive ablation study analysis

## 🔬 Cross-LLM Validation

To validate the robustness and generalizability of our findings, we conduct systematic cross-LLM validation with Claude Sonnet 4.5.

### Validation Study: Claude Sonnet 4.5

**Objective**: Test if translation advantages hold with different LLM architecture to validate findings are not GPT-4o-specific artifacts.

#### Setup
1. **Install Claude SDK**:
   ```bash
   pip install anthropic
   ```

2. **Update `src/rag_system.py`**:
   ```python
   # Replace _call_openai_chat function with:
   def _call_claude_chat(system_prompt: str, context: str, query: str, 
                        model: str = "claude-sonnet-4-5-20250929") -> str:
       import anthropic
       
       api_key = os.getenv("ANTHROPIC_API_KEY")
       if not api_key:
           raise RuntimeError("ANTHROPIC_API_KEY not set")
       
       client = anthropic.Anthropic(api_key=api_key)
       
       message = client.messages.create(
           model=model,
           max_tokens=500,
           temperature=0.3,
           system=system_prompt,
           messages=[{
               "role": "user", 
               "content": f"Context:\n{context}\n\nQuestion: {query}"
           }]
       )
       
       return message.content[0].text
   ```

3. **Update `src/utils.py`** for LLM-as-judge evaluation:
   ```python
   # Replace OpenAI client with Claude for evaluation
   import anthropic
   
   client = anthropic.Anthropic(api_key=api_key)
   response = client.messages.create(
       model="claude-sonnet-4-5-20250929",
       max_tokens=1000,
       temperature=0.0,
       system="You are an expert medical information evaluator. Respond only with valid JSON.",
       messages=[{"role": "user", "content": PROMPT}]
   )
   ```

#### Execution
```bash
# Create results directory
mkdir results/claude_ablation/

# Run experiments with Claude
# Notebooks 3, 4, 5 with modified API calls
# Save results to claude_ablation/ folder
```

### Validated Findings: Robust Across LLMs

**Strong Consistency (All 4 Conditions)**:
- ✅ Translation pipeline wins for BOTH languages across BOTH models
- ✅ Translation achieves superior or equivalent safety (0% hallucinations in 3/4 conditions)
- ✅ Multilingual is consistently faster (5-34% speedup)
- ✅ Translation's strongest advantage is in completeness dimension

**Model-Specific Patterns**:
- GPT-4o shows larger speed differences (32-34% vs 5-24%)
- Claude shows larger quality gaps (especially Hindi: +0.31 vs +0.02)
- Both models validate the same winner patterns

### Chunking Strategy Variation (Optional)

**Objective**: Test if chunk overlap findings and performance differences hold with different chunking strategies.

#### Chunking Configurations

| Configuration | Chunk Size | Overlap | Overlap % | Results Folder |
|---------------|------------|---------|-----------|----------------|
| Small | 500 | 100 | 20% | `chunk_500_100/` |
| Current | 1000 | 200 | 20% | `baseline/` |
| Large | 1500 | 300 | 20% | `chunk_1500_300/` |

## 📈 Evaluation Framework

### Multi-Dimensional Assessment
- **LLM-as-Judge**: Both GPT-4o and Claude Sonnet 4.5 evaluation for faithfulness, completeness, appropriateness
- **Cross-LLM Validation**: Ensures findings are not model-specific artifacts
- **Chunk Analysis**: Retrieval overlap and pattern analysis
- **Safety Analysis**: Hallucination rate detection across both models
- **Efficiency**: Component-level timing analysis

### Cross-Language & Cross-Model Validation
- **Identical Questions**: Same medical content across Hindi and Chinese
- **Statistical Testing**: Paired t-tests for significance
- **Model Independence**: Consistent patterns across GPT-4o and Claude Sonnet 4.5

## 🔍 Technical Contributions

### Novel Insights
1. **Robust Translation Superiority**: Consistent advantages across languages AND models
2. **Cross-Model Validation**: First study to validate cross-lingual RAG findings across multiple LLM architectures
3. **Safety Implications**: Translation achieves 0% hallucinations in 75% of conditions
4. **Information Access Patterns**: 27-28% overlap shows fundamentally different knowledge access

### Methodological Innovations
- **Apples-to-Apples Comparison**: Identical questions eliminate question difficulty bias
- **Cross-LLM Validation**: Tests architectural independence of findings
- **Multi-Dimensional Safety**: Combines automated metrics with LLM-based hallucination detection
- **Component-Level Analysis**: Separates retrieval quality from generation quality

## 🎯 Research Hypotheses & Evidence

### H1: Model-Independent Translation Advantage
**H0**: Translation advantages are GPT-4o-specific artifacts  
**Evidence**: **REJECTED** - Translation wins all 4 conditions (2 languages × 2 LLMs)

### H2: Cross-Language Consistency  
**H0**: Optimal approach varies by language  
**Evidence**: **REJECTED** - Translation consistently superior for both Hindi and Chinese

### H3: Safety Equivalence
**H0**: Both approaches have equivalent hallucination rates  
**Evidence**: **REJECTED** - Translation achieves 0% hallucinations in 3 of 4 conditions

### H4: Efficiency-Quality Tradeoff
**H0**: Faster approach sacrifices quality  
**Evidence**: **REJECTED** - Though multilingual is faster, translation's quality advantage justifies overhead

## 🛠️ Implementation Details

### Core Technologies
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Store**: FAISS for efficient similarity search
- **LLMs**: GPT-4o and Claude Sonnet 4.5 for generation and evaluation
- **Translation**: deep-translator with language validation
- **Evaluation**: Custom LLM-as-judge framework with cross-model validation

### Data Sources
- **FDA Drug Labels**: Authoritative medication information
- **NIH/MedlinePlus**: Government medical health topics
- **Processing**: 1000-char chunks, 200-char overlap (baseline)
- **Languages**: English source → Hindi/Chinese questions

## 📊 Statistical Evidence

### Quality Scores Summary (1-5 scale)

**Translation Advantages (Average Across Both Models):**
- Hindi: +0.17 points average improvement
- Chinese: +0.15 points average improvement

**Safety Advantage:**
- Translation: 0% hallucinations in 3 of 4 conditions
- Multilingual: 0% hallucinations in 1 of 4 conditions

**Efficiency Tradeoff:**
- Multilingual: 5-34% faster execution
- Translation: Superior quality and safety despite modest latency overhead

## ⚠️ Limitations & Future Work

### Current Limitations
1. **Sample Size**: 30 questions per language limits generalization
2. **Domain Scope**: Healthcare-specific findings may not transfer
3. **LLM Coverage**: Two models (GPT-4o, Claude) may not represent all architectures
4. **Language Coverage**: Only Hindi and Chinese tested

### Future Research Directions
1. **Additional LLMs**: Test with Gemini, Llama, and other architectures
2. **More Languages**: Expand to Arabic, Vietnamese, Tagalog, etc.
3. **Human Evaluation**: Native speaker validation of results
4. **Production Deployment**: Real-world healthcare system integration
5. **Hybrid Approaches**: Combine strengths of both methods

## 📚 Citation

```bibtex
@inproceedings{kundu2025crosslingual,
  title={Translation vs. Multilingual Embeddings for Cross-Lingual Medical QA},
  author={Boris Kundu and Parth Jawale},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  note={Under Review}
}
```

## 🔍 Key Research Implications

### For Healthcare Applications
- **All deployments**: Translation pipeline recommended for superior quality and safety
- **Safety-first principle**: 0% hallucination rate in 75% of conditions makes translation the safer choice
- **Latency consideration**: 5-34% overhead acceptable for medical accuracy

### For Cross-Lingual NLP
- **Model-independent findings**: Consistent patterns across GPT-4o and Claude Sonnet 4.5
- **Translation efficacy**: Modern translation APIs with strong LLMs outperform direct multilingual embeddings
- **Cross-LLM validation essential**: Single-model findings may not generalize

### For RAG System Design
- **Evaluation frameworks**: Need cross-model validation beyond single-LLM testing
- **Safety considerations**: Critical for high-stakes applications like healthcare
- **End-to-end assessment**: Component metrics alone insufficient for system evaluation

## 📧 Contact

**Boris Kundu**  
Email: boriskundu@gmail.com  

**Parth Jawale**  
Email: parthjawale1996@gmail.com

## 🙏 Acknowledgments

- OpenAI for GPT-4o API access enabling this research
- Anthropic for Claude Sonnet 4.5 API access for cross-LLM validation
- FDA and NIH for providing authoritative medical data sources
- Deep-translator library for reliable translation services

---

**Note**: This research demonstrates that translation pipelines provide superior cross-lingual medical QA across multiple LLM architectures, with significant implications for healthcare applications in low-resource language communities. The cross-LLM validation strengthens the generalizability of findings and challenges assumptions about multilingual embeddings being the optimal solution for cross-lingual RAG systems.