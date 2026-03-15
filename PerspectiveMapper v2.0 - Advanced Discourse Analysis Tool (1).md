# PerspectiveMapper v2.0 - Advanced Discourse Analysis Tool

A comprehensive Streamlit application for analyzing discourse, documents, and text corpora with advanced NLP techniques including topic modeling, sentiment analysis, bias detection, and document similarity analysis.

## ✨ Key Features

### 📁 Multi-Format File Support
- **Text Files** (.txt) - Plain text documents
- **Word Documents** (.docx) - Microsoft Word files
- **PDF Files** (.pdf) - Portable Document Format
- **Excel Files** (.xlsx, .xls) - Spreadsheet data extraction

### 📊 Analysis Capabilities

#### Document Statistics
- Character and word counts
- Vocabulary diversity metrics
- Average word length
- Sentence count analysis

#### Topic Modeling (LDA)
- Latent Dirichlet Allocation with configurable topics
- Top words per topic extraction
- Topic dominance per document
- Adjustable vocabulary size

#### Document Clustering
- Sentence-BERT embeddings for semantic understanding
- PCA dimensionality reduction
- K-Means clustering with visualization
- Hierarchical clustering dendrogram

#### Similarity Analysis
- Cosine similarity matrix between documents
- Heatmap visualization
- Semantic document relationships

#### Sentiment Analysis
- **CardiffNLP** (XLM-RoBERTa based) - Multilingual support
- **VADER** (fallback) - Rule-based sentiment
- Sentiment distribution visualization
- Per-document sentiment scores

#### Bias Detection
- Customizable bias keyword dictionaries
- Multi-category bias analysis (gender, migration, religion, politics)
- Bias score calculation per document
- Visual bias distribution charts

#### TF-IDF Analysis
- Term Frequency-Inverse Document Frequency scoring
- Top terms identification
- Relative importance metrics

### 🔐 Security
- Password-protected access via Streamlit secrets
- Session-based authentication

### 📥 Export Options
- **CSV Export** - Results in tabular format
- **JSON Export** - Structured data with metadata
- **Visualizations** - Interactive Plotly charts

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone or download the project**
```bash
cd perspective_mapper
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Streamlit secrets** (optional, for password protection)

Create `.streamlit/secrets.toml`:
```toml
[passwords]
admin = "your_secure_password"
user1 = "password123"
```

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Basic Workflow

1. **Upload Documents**
   - Use the sidebar file uploader
   - Supports multiple files simultaneously
   - Accepts .txt, .docx, .pdf, .xlsx formats

2. **Configure Analysis**
   - Select stopword languages
   - Add custom stopwords
   - Choose visualization options
   - Set model parameters (topics, clusters, vocabulary size)

3. **Run Analysis**
   - Application automatically processes documents
   - Generates statistics, visualizations, and metrics
   - Displays results in organized sections

4. **Export Results**
   - Download CSV for spreadsheet analysis
   - Download JSON for programmatic access
   - Save visualizations as images

## ⚙️ Configuration Options

### Language & Preprocessing
- **Stopword Languages**: Multi-language support (EN, ES, IT, FR, DE, PT, NL, RU, AR)
- **Extra Stopwords**: Custom stopword addition
- **Min Token Length**: Automatically filters tokens < 3 characters

### Model Parameters
- **LDA Topics**: 2-12 topics (default: 5)
- **Max Vocabulary**: 1,000-10,000 terms (default: 3,000)
- **KMeans Clusters**: 2-12 clusters (default: 4)

### Sentiment Analysis
- **CardiffNLP**: XLM-RoBERTa multilingual model (recommended)
- **VADER**: Fallback rule-based analyzer

### Bias Detection
- **Customizable Keywords**: JSON-based bias category definitions
- **Default Categories**: Gender, Migration, Religion, Politics
- **Flexible Extension**: Add custom categories as needed

## 📊 Output Sections

### 1. Document Statistics
- Raw and cleaned text metrics
- Vocabulary analysis
- Language detection

### 2. WordClouds
- Visual term frequency representation
- Per-document analysis
- Customizable display

### 3. TF-IDF Analysis
- Top 20 terms by importance
- Average TF-IDF scores
- Bar chart visualization

### 4. Topic Modeling (LDA)
- Topic-term relationships
- Dominant topic per document
- Topic coherence metrics

### 5. Clustering Visualization
- 2D PCA projection
- K-Means cluster assignments
- Interactive scatter plot

### 6. Hierarchical Clustering
- Dendrogram visualization
- Document relationship tree
- Distance-based grouping

### 7. Similarity Matrix
- Pairwise cosine similarity
- Heatmap visualization
- Document relationship strength

### 8. Sentiment Analysis
- Per-document sentiment labels
- Confidence scores
- Distribution charts

### 9. Bias Analysis
- Bias keyword frequency
- Multi-category analysis
- Normalized bias scores

### 10. Comprehensive Results Table
- All metrics consolidated
- Sortable and filterable
- Export-ready format

## 🔧 Advanced Features

### Custom Bias Categories

Edit the bias dictionary JSON in the sidebar:

```json
{
  "gender": ["woman", "man", "trans", "equality"],
  "migration": ["immigrant", "migrant", "refugee"],
  "custom_category": ["keyword1", "keyword2"]
}
```

### Multilingual Support

The application automatically detects document language and applies appropriate:
- Stopwords (via NLTK)
- Tokenization rules
- Sentiment analysis models

### Embedding Models

The application uses `paraphrase-multilingual-MiniLM-L12-v2` for semantic embeddings:
- Supports 50+ languages
- Fast inference
- High-quality representations

## 📈 Performance Considerations

- **Large Documents**: Process in batches if > 10MB total
- **Many Documents**: Clustering may be slow with > 100 documents
- **Model Loading**: First run downloads models (~500MB)
- **Memory**: Requires ~4GB RAM for typical analysis

## 🐛 Troubleshooting

### PDF Reading Issues
- Ensure pdfplumber is installed: `pip install pdfplumber`
- Some encrypted PDFs may not be readable
- Try converting to text format if issues persist

### Memory Errors
- Reduce max vocabulary size
- Process fewer documents at once
- Increase system RAM or use cloud deployment

### Slow Performance
- Disable CardiffNLP if not needed (use VADER)
- Reduce number of topics/clusters
- Use smaller vocabulary size

### Language Detection Errors
- Ensure documents have sufficient text (> 100 characters)
- Check language is in supported list
- Manually specify language in stopwords

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web framework |
| scikit-learn | ML algorithms (LDA, KMeans, PCA) |
| sentence-transformers | Semantic embeddings |
| transformers | CardiffNLP sentiment model |
| plotly | Interactive visualizations |
| matplotlib | Static visualizations |
| python-docx | DOCX file reading |
| openpyxl | Excel file reading |
| pdfplumber | PDF file reading |
| nltk | Stopwords and NLP utilities |
| pandas | Data manipulation |
| numpy | Numerical computing |

## 🔐 Security Notes

- Store passwords in `.streamlit/secrets.toml` (not in code)
- Never commit secrets file to version control
- Use strong passwords for production deployments
- Consider OAuth integration for enterprise use

## 📝 Example Workflows

### Academic Research
1. Upload research papers (PDF)
2. Extract topics and themes
3. Analyze sentiment and bias
4. Export results for literature review

### Social Media Analysis
1. Upload text exports from platforms
2. Detect sentiment trends
3. Identify bias in discourse
4. Cluster similar discussions

### Content Analysis
1. Upload articles/documents
2. Analyze topic distribution
3. Compare document similarity
4. Identify key terms (TF-IDF)

### Policy Analysis
1. Upload policy documents
2. Detect bias language
3. Compare policy documents
4. Identify key themes

## 🚀 Deployment

### Streamlit Cloud
```bash
streamlit run app.py
# Then push to GitHub and connect to Streamlit Cloud
```

### Docker
```dockerfile
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Environment Variables
```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=info
```

## 📄 License

MIT License - Feel free to use and modify

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional language support
- More sentiment models
- Custom clustering algorithms
- Real-time streaming analysis
- Database integration

## 📧 Support

For issues, questions, or suggestions:
1. Check the Troubleshooting section
2. Review example workflows
3. Check dependency versions
4. Consult Streamlit documentation

## 🎓 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Sentence Transformers](https://www.sbert.net/)
- [Plotly](https://plotly.com/)
- [NLTK Documentation](https://www.nltk.org/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)

---

**PerspectiveMapper v2.0** - Making discourse analysis accessible and comprehensive.
