# AI Resume Screening & Recruitment Agent

An AI-powered recruitment system that screens resumes against a Job Description and helps recruiters analyze and compare candidates.

## Technologies Used

### Programming Language

* Python

### AI / Machine Learning

* Machine Learning
* Natural Language Processing (NLP)
* TF-IDF
* Sentence Transformers
* Semantic Similarity

### Python Libraries

* Streamlit
* Pandas
* NumPy
* Scikit-learn
* pdfplumber
* python-docx
* ReportLab
* Sentence Transformers

### Tools

* Git
* GitHub
* Visual Studio Code

## Features

* Upload Job Description
* Upload multiple PDF and DOCX resumes
* Extract text from resumes
* Match candidate skills with Job Description requirements
* Calculate text similarity
* Calculate semantic similarity
* Generate candidate scores
* Rank candidates
* Identify candidate skill gaps
* Filter and search candidates
* Track recruiter status
* Generate PDF reports
* AI Recruitment Agent for candidate analysis and comparison
* Generate interview questions

## Working Flow

```text
Job Description
       ↓
Extract Required Skills
       ↓
Upload Resumes
       ↓
Resume Parsing
(PDF / DOCX)
       ↓
Text Preprocessing
       ↓
Skill Matching
       ↓
TF-IDF Text Similarity
       ↓
Sentence Transformer
Semantic Similarity
       ↓
Final Candidate Score
       ↓
Candidate Ranking
       ↓
AI Recruitment Agent
       ↓
Recruiter Decision Support
```

## How Candidate Scoring Works

The system calculates the final score using three components:

```text
Final Score =
50% Skill Matching
+
30% Semantic Similarity
+
20% Text Similarity
```

### 1. Skill Matching

The system compares the skills required in the Job Description with the skills found in the candidate's resume.

### 2. TF-IDF Text Similarity

TF-IDF is used to compare the important words and terms in the Job Description and resume.

### 3. Semantic Similarity

Sentence Transformers convert the Job Description and resume into numerical embeddings and compare their meaning using similarity.

This helps identify candidates even when they use different words with similar meanings.

## AI Recruitment Agent

The AI Recruitment Agent works on top of the screening results and provides recruiter-focused analysis.

It can:

* Find candidates
* Rank candidates
* Compare candidates
* Identify missing skills
* Explain candidate results
* Provide recruitment insights
* Generate interview questions

For example:

```text
Who is the highest-scoring candidate?

Compare Rahul and Priya.

What skills is Rahul missing?

Show candidates with Java and SQL.

Generate interview questions for this candidate.
```

## Project Working

The application is built using **Streamlit**, which provides the user interface.

When a recruiter uploads a Job Description and resumes:

1. The system reads the Job Description.
2. Resumes are parsed from PDF or DOCX.
3. Resume text is cleaned and preprocessed.
4. Required skills are compared with candidate skills.
5. TF-IDF calculates text similarity.
6. Sentence Transformers calculate semantic similarity.
7. These results are combined into a final candidate score.
8. Candidates are ranked based on their scores.
9. The AI Recruitment Agent uses these results to answer recruiter queries and provide decision support.

## Project Structure

```text
AI-Resume-Screening-Recruitment-Agent/
│
├── agent/
│   ├── agent_tools.py
│   └── recruitment_agent.py
│
├── data/
│   ├── job_description.txt
│   ├── resume1.pdf
│   ├── resume2.pdf
│   └── resume3.docx
│
├── models/
│   ├── candidate_info.py
│   ├── semantic_similarity.py
│   ├── similarity.py
│   └── skill_matcher.py
│
├── parser/
│   └── resume_parser.py
│
├── preprocessing/
│   └── text_preprocessing.py
│
├── tests/
│   └── test_agent.py
│
├── app.py
├── requirements.txt
└── .gitignore
```

## Run the Project

Clone the repository:

```bash
git clone <your-repository-url>
cd AI-Resume-Screening-Recruitment-Agent
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in the browser.
