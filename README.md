## 🧘‍♂️ **Equinox – AI-Powered Health & Wellness Analyzer**

> *“Balancing minds, one insight at a time.”*

---

### 🌍 **Overview**

**Equinox** is an AI-driven web application that analyzes social media text (Reddit posts) to generate **emotional insights, personality profiles, and wellness indicators**.
It visualizes trends like **mood evolution, aura clusters, and emotion distributions** using cutting-edge NLP models such as **Sentence-BERT**, **DistilRoBERTa**, and **VADER**.

The platform offers:

* 🌈 **Aura detection** (emotional clustering via K-Means)
* 📈 **Mood & emotion visualization** dashboards
* 🧠 **Personality insights** using the **Big Five (OCEAN)** model
* 💬 **Therapist dashboard** for patient search & analysis
* ❤️ **Mental-health risk warnings** (depression, anxiety, PTSD, schizophrenia)

---

### 🧩 **Architecture Overview**

```
┌─────────────────────────────────────┐
│   Reddit API (PRAW)                │  ← Data source
└──────────────┬──────────────────────┘
               │
               ▼
   ┌──────────────────────────────┐
   │  Flask Backend (app.py)      │
   │  + REST APIs (/api/*)        │
   └──────────────┬───────────────┘
                  │
     ┌──────────────────────────────┐
     │  NLP Models                  │
     │  • VADER (sentiment)         │
     │  • Sentence-BERT (embeddings)│
     │  • K-Means (aura clusters)   │
     │  • DistilRoBERTa (emotions)  │
     │  • Big Five (OCEAN)          │
     └──────────────┬───────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  Frontend (HTML + JS)│
        │  • User Dashboard    │
        │  • Therapist Portal  │
        └──────────────────────┘
```

---

### ⚙️ **Installation & Setup**

#### 🧾 Prerequisites

* Python ≥ 3.10
* pip (latest version)
* Reddit API credentials

#### 🪜 Steps

```bash
# 1️⃣ Clone the repository
git clone https://github.com/ombansal/equinox.git
cd equinox

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add Reddit API credentials
#    Inside app.py → replace:
#    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# 4️⃣ Run the Flask app
python app.py
```

Then open → **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

### 💡 **Core Workflow**

| Stage                        | Description                                     | Tools / Models                   |
| ---------------------------- | ----------------------------------------------- | -------------------------------- |
| **1. Data Collection**       | Scrape Reddit posts using PRAW                  | `ingest.py`                      |
| **2. Pre-processing**        | Clean & normalize text (lowercase, remove URLs) | Regex / Pandas                   |
| **3. Sentiment Analysis**    | Compute compound scores                         | VADER                            |
| **4. Embedding Generation**  | Create vector representations                   | Sentence-BERT (all-MiniLM-L6-v2) |
| **5. Clustering**            | Identify aura groups                            | K-Means                          |
| **6. Emotion Detection**     | Detect fine-grained emotions                    | DistilRoBERTa                    |
| **7. Personality Profiling** | Map linguistic features to OCEAN traits         | Big Five Model                   |
| **8. Visualization**         | Dashboards + Charts                             | Chart.js, Matplotlib             |

---

### 🧠 **Models Used**

| Model                | Type                | Purpose                                 |
| -------------------- | ------------------- | --------------------------------------- |
| **Naive Bayes**      | Probabilistic       | Baseline sentiment classification       |
| **VADER**            | Rule-based          | Social-media sentiment detection        |
| **Sentence-BERT**    | Transformer         | Semantic embeddings of text             |
| **K-Means**          | Unsupervised ML     | Emotional clustering / Aura detection   |
| **Big Five (OCEAN)** | Psychological Model | Personality trait prediction            |
| **DistilRoBERTa**    | Transformer         | Emotion classification (seven emotions) |

---

### 🧩 **Folder Structure**

```
equinox/
│
├── app.py                    # Flask backend
├── aura.py                   # Aura clustering logic
├── nlp_bert.py               # Emotion analysis (DistilRoBERTa)
├── ingest.py                 # Reddit scraper + cache
│
├── templates/                # HTML templates
│   ├── home.html
│   ├── dashboard.html
│   ├── therapist_dashboard.html
│   └── patient_detail.html
│
├── static/                   # Static assets (CSS / Images)
│   └── style.css
│
└── uploads/                  # Cached Reddit data (JSON / CSV)
```

---

### 📊 **Example Visualizations**

* **Mood Trend Graph:** Average VADER sentiment per day
* **Emotion Radar:** Weekly emotion distribution from BERT
* **Aura Card:** User’s dominant emotional cluster
* **Therapist Dashboard:** Patient search & dominant emotion summary

---

### 🔐 **Reddit API Credentials**

Create a Reddit app at [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
and update `app.py`:

```python
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "Equinox-App-by-Om-Bansal"
```

---

### 🚀 **Future Enhancements**

* Integrate real-time emotion updates
* Expand to Twitter/Instagram data
* Add multi-language support
* Deploy on Azure App Service / AWS EC2

---

### 👤 **Author**

**Om Bansal**
📧 [[ombansal@example.com](mailto:ombansal2109@gmail.com)]
💼 [LinkedIn](https://linkedin.com/in/om~bansal/) | [GitHub](https://github.com/ombansal1)

---
