# 📝 Google Maps Review Sentiment Analysis & Regional Insights for Bakauheni Harbour City (BHC)

This project focuses on processing, sentiment classification, keyword extraction, topic modeling, and delivering interactive visual insights alongside AI-generated strategic recommendations based on user reviews scraped from **Google Maps** in the Bakauheni area, South Lampung.

---

## 🗺️ Background & Sector/District Allocation

The spatial analysis and mapping of retail venues, facilities, and amenities within this dataset are categorized into 3 primary development sectors of the national strategic project:
1. **Harbourfront and Intermoda Area**
2. **Hilltop Resort**
3. **Marina District**

### 📌 Sector Nomenclature Reference Source
The naming of these sectors is officially based on the macro planning and integrated tourism zoning blueprint from the **Bakauheni Harbour City (BHC) Masterplan**, initiated by **PT ASDP Indonesia Ferry (Persero)** in collaboration with key corporate stakeholders (*Hutama Karya, ITDC*). 

Geographic delineation boundaries (leveraging the `place_lat` and `place_lng` coordinates) are used to classify retail stores, local shops (*toserba*), tourist spots, and port infrastructure into their respective planned commercial clusters to measure the actual, live customer experience on the ground.

---

## ⚙️ Workflow & Data Pipeline Architecture

The project consists of a core two-part technical implementation: a backend data engineering pipeline and a frontend interactive analytics dashboard.

+------------------------------------------+      +------------------------------------------+
|          1. BACKEND NLP PIPELINE         |      |        2. FRONTEND STREAMLIT APP         |
|        (sentiment_analyst.ipynb)         |      |          (app-dashboard.py)              |
+------------------------------------------+      +------------------------------------------+
|  [Start: Read Raw Google Maps Data]      |      |  [Load Pre-processed df_final Data]      |
|                  │                       |      |                  │                       |
|                  ▼                       |      |                  ▼                       |
|  [Data Cleaning & Text Validation]       |      |  [Dynamic Sector & Sentiment Filters]    |
|                  │                       |      |                  │                       |
|                  ▼                       |      |                  ▼                       |
|  [Sentiment Classification (IndoBERT)]   |----> |  [Interactive Plotly Renderings]         |
|                  │                       |      |  - Sentiment Distribution Metrics        |
|                  ▼                       |      |  - Topic Cluster & Sentiment Overviews   |
|  [Keyword Extraction via KeyBERT]        |      |                  │                       |
|                  │                       |      |                  ▼                       |
|                  ▼                       |      |  [Deep Dive: Store-level AI Advice]      |
|  [Topic Modeling using BERTopic]         |      |                  │                       |
|                  │                       |      |                  ▼                       |
|                  ▼                       |      |  [Filtered Sub-dataset Export (.csv)]    |
|  [Venue Aggregation & AI Recommendations]|      +------------------------------------------+
|                  │                       |
|                  ▼                       |
|  [End: Export to df_final CSV/JSON]      |
+------------------------------------------+


### 🔍 Detailed Processing & Component Modules
1. **Backend Pipeline (`sentiment_analyst.ipynb`):**
   * **Data Ingestion & Cleaning:** Validation of geographical schemas and textual review normalization (handling common Indonesian slang/typos and eliminating emojis).
   * **Classifier (IndoBERT):** Polarity detection utilizing an Indonesian-optimized, Transformer-based language model to classify reviews into **Positive, Negative, or Neutral**.
   * **Keyword & Topic Extraction:** Utilizing **KeyBERT** (`indobenchmark/indobert-base-p2`) and **BERTopic** to map core structural pain points or praises by sector automatically.
   * **AI Recommendation:** Passing aggregated sentiment profiles per vendor into an LLM to synthesize actionable operational recommendations.

2. **Frontend UI Dashboard (`app-dashboard.py`):**
   * **Interactive Navigation:** Implements custom CSS injection with an Inter font aesthetic styling layer and a hero section backdrop (`pel_bakauheni.jpg`).
   * **Granular Analytical Filters:** Syncs sector selectors (`ALL SECTOR`, `Harbourfront and Intermoda Area`, etc.) with score thresholds and query search bars.
   * **Visualization & Export Suite:** Generates dynamic **Plotly Express** graphical distributions and renders structural layout tables with localized data down-sampling (`.csv` export button triggered dynamically).

---

## 📊 Data Schema & Feature Descriptions

The final generated `df_final` dataset contains the following structural attributes:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `place_name` | String / Category | Scraped name of the commercial outlet, retail shop, or facility in Bakauheni. |
| `place_lat` | Float (Coordinate) | Latitude physical coordinate map marker of the venue. |
| `place_lng` | Float (Coordinate) | Longitude physical coordinate map marker of the venue. |
| `sector` | Category | BHC macro sector class matching the coordinates (*Harbourfront, Hilltop Resort, Marina District*). |
| `reviewer_name` | String | Public display name of the Google Maps reviewer. |
| `rating` | Integer (1 - 5) | Star rating score scales given by the reviewer from 1 to 5. |
| `time` | String / Category | Textual timestamp stating when the review was posted (e.g., "1 month ago", "2 years ago"). |
| `text` | String | Raw, untampered text review fetched from the Google Maps scraping process. |
| `text_clean` | String | Fully cleaned text string prepared for core NLP model inference. |
| `sentiment` | Category | Model-predicted sentiment label outcome: **Positive / Negative / Neutral**. |
| `score` | Float (0.0 - 1.0) | Model confidence metric for the predicted sentiment classification label. |
| `rekomendasi_ai` | String | Automated, AI-generated strategic advice focusing on target improvements or marketing hooks for the venue. |

---

## 🛠️ Technology Stack & Libraries

* **Machine Learning & Data Infra:** `torch`, `transformers`, `pandas`, `numpy`, `tqdm`
* **NLP Extensions:** `keybert`, `bertopic`, `sentence-transformers`
* **Visualization & Deployment Dashboard:** `streamlit`, `plotly`, `base64`, `re`, `pathlib`
* **Generative AI:** `groq API SDK (model: llama-3.3-70b-versatile)`

---

## 💡 Key Dashboard Insight Features

* **Sector Metrics Overview:** Visualizes spatial and thematic distribution shifts across the BHC masterplan layout (e.g., tracking operational queues in the *Harbourfront and Intermoda Area* vs. hospitality feedback scores in the *Hilltop Resort* zone).
* **AI Recommendation Engine Hub:** Displays structural data tables detailing store-level customer service slips or infrastructure drawbacks, translating unstructured review feedback into concrete operational checkpoints.

---

### 🚀 Getting Started (How to Run)

#### 1. Backend Feature Generation
Run the core pipeline (`sentiment_analyst.ipynb`) inside a GPU runtime environment (e.g., Google Colab) to output the fully structured aggregated analytical dataset file.

#### 2. Launching the Streamlit Web UI Dashboard
Ensure your file structure includes the visual asset backdrop image (`pel_bakauheni.jpg`) inside the root folder path, then initialize the local deployment instance via terminal execution:

```bash
# Install dependencies
pip install streamlit plotly pandas

# Run the UI instance
streamlit run app-dashboard.py