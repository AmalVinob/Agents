# 🧬 Breast Cancer AI Assistant - Multi-Agent Diagnostic System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange.svg)
![Groq](https://img.shields.io/badge/Groq-AI-purple.svg)

A cutting-edge AI-powered medical assistant that uses parallel multi-agent architecture to provide comprehensive breast cancer risk assessment and diagnostic recommendations. The system integrates genomic analysis, clinical history evaluation, and medical imaging analysis to deliver personalized treatment plans in hours instead of weeks.

## 🚀 Key Features

- **⚡ Parallel Multi-Agent Processing**: Simultaneous analysis by specialized AI agents
- **🧬 Genomic Variant Analysis**: BRCA1 only mutation detection and risk assessment
- **📋 Clinical History Matching**: Comprehensive patient profile analysis
- **🖼️ Medical Imaging Analysis**: CNN-based histopathology image classification
- **📊 Consolidated Reporting**: Unified diagnostic recommendations
- **⏱️ Rapid Processing**: Complete analysis in hours vs. traditional weeks/months

## 🏗️ Architecture

```
Patient Data Input
        │
   ┌────┼────┐
   │    │    │
Genomic Imaging Clinical
Agent   Agent   Agent
   │    │    │
   └────┼────┘
        │
Synthesized Report
(Personalized Plan)
```

### Multi-Agent System Components

1. **🧬 Genomic Variant Analysis Agent**
   - Processes VCF/CSV genomic data
   - Identifies pathogenic mutations
   - Calculates hereditary cancer risk scores
   - Focuses on BRCA1/BRCA2 variants

2. **📋 Clinical History Matching Agent**
   - Analyzes patient demographics and symptoms
   - Processes clinical history data
   - Evaluates risk factors and comorbidities
   - Provides clinical context for genomic findings

3. **🖼️ Imaging Analysis Agent**
   - CNN-based histopathology image analysis
   - Fine-tuned MobileNetV2 architecture
   - Breast tissue classification
   - Confidence scoring for radiologist review
  

## 📸 Screenshots

### Main Interface
*Multi-agent analysis dashboard showing parallel processing of genomic, imaging, and clinical data*

### Genomic and ROI Image Analysis 
![Genomic Analysis](https://github.com/AmalVinob/Agents/blob/main/LangGraph/Breast%20Cancer%20AI%20Assistant/Screenshot%20(28).png?raw=true)
*BRCA1 variant analysis with pathogenicity scoring and risk assessment*
*CNN-based histopathology image classification with confidence scores* 

### Clinical Profile Input
![Clinical Profile](https://github.com/AmalVinob/Agents/blob/main/LangGraph/Breast%20Cancer%20AI%20Assistant/Screenshot%20(29).png?raw=true )
*Patient data input interface for clinical history and demographic information*

### Consolidated Report
![Consolidated Report](https://github.com/AmalVinob/Agents/blob/main/LangGraph/Breast%20Cancer%20AI%20Assistant/Screenshot%20(30).png?raw=true )
*AI-generated comprehensive diagnostic report with personalized recommendations*


## 📊 Datasets Used

| Dataset | Source | Purpose |
|---------|--------|---------|
| **1000 Genomes Project** | [EBI FTP](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/) | Population genomics reference |
| **ClinVar** | [NCBI ClinVar](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr17.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz) | Variant-disease associations |
| **Breast Histopathology Images** | [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images) | Medical imaging training data |

## 🛠️ Technology Stack

- **Framework**: LangGraph for multi-agent orchestration
- **LLM**: Groq API (Gemma2-9B-IT model)
- **Deep Learning**: TensorFlow/Keras with MobileNetV2
- **Pre-trained Weights**: ImageNet (fine-tuned)
- **Data Processing**: Pandas, NumPy, BioPython
- **Visualization**: Matplotlib, Seaborn

## 📋 Prerequisites

- Python 3.8+
- Groq API Key ([Get yours here](https://groq.com/))
- CUDA-compatible GPU (recommended for imaging analysis)
- Minimum 8GB RAM

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/breast-cancer-ai-assistant.git
cd breast-cancer-ai-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
cp .env.example .env

# Add your API key to .env file
GROQ_API_KEY=your_groq_api_key_here
```

## How to Run the code
- CNN/preprocessing
```bash
python preprocessing.py
```
- Run main/personalized_agent.py
```bash
streamlit run personalized_agent.py
```

## 🔬 Model Details

### Genomic Analysis
- Variant calling pipeline with GATK standards
- Machine learning classification for pathogenicity
- Population frequency analysis from 1000 Genomes

### Imaging Model
- **Architecture**: MobileNetV2 (fine-tuned)
- **Input Size**: 224x224x3
- **Classes**: Normal, Benign, Malignant
- **Training**: Transfer learning from ImageNet

## 🚨 Important Disclaimers

⚠️ **MEDICAL DISCLAIMER**: This software is for research and educational purposes only. It is NOT intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

⚠️ **DATA PRIVACY**: Ensure all patient data is properly anonymized and handled according to HIPAA and other applicable regulations.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/AmalVinob/Agents/blob/main/LICENSE) file for details.

## 🙏 Acknowledgments

- **1000 Genomes Project** for providing open genomic data
- **ClinVar** for variant-disease association database
- **Kaggle Community** for the breast histopathology dataset
- **Groq** for high-performance LLM inference
- **LangGraph** team for the multi-agent framework

## 📞 Contact

- **Developer**: Amal Vinob
- **Email**: amalvinob662@gmail.com



## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{breast_cancer_ai_assistant,
  title={Breast Cancer AI Assistant: Multi-Agent Diagnostic System},
  author={Amal Vinob},
  year={2025},
  url={https://github.com/yourusername/breast-cancer-ai-assistant}
}
```

---

**Built with ❤️ for advancing personalized medicine and improving patient outcomes.**
