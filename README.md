# Thesis
# Vietnamese Legal Question Answering Chatbot using LangChain and RAG

## 1. Project Overview

This project implements a Vietnamese legal question-answering chatbot based on the Retrieval-Augmented Generation (RAG) framework. The system is designed to retrieve relevant information from Vietnamese legal documents and generate grounded answers using a Large Language Model (LLM).

The project focuses on Vietnamese legal documents related to grassroots mediation, particularly Resolution `01/2014/NQLT/CP-UBTƯMTTQVN`. It compares different retrieval sources and retrieval strategies to evaluate their effects on answer quality, retrieval accuracy, and response latency.

The implementation uses LangChain, Chroma vector database, Gemini embeddings, Gemini Flash, and Gradio for interactive demonstration.

---

## 2. Main Objectives

The main objectives of this project are:

* To build an end-to-end RAG-based chatbot for Vietnamese legal question answering.
* To preprocess and structure Vietnamese legal documents into retrievable document objects.
* To construct two retrieval sources: a legal content corpus and a question-answer corpus.
* To compare retrieval performance between content-based retrieval and QA-based retrieval.
* To evaluate chatbot responses using lexical, semantic, retrieval-based, and latency-based metrics.
* To develop an interactive Gradio interface for demonstration and analysis.

---

## 3. System Architecture

The proposed system follows a Retrieval-Augmented Generation pipeline consisting of the following components:

1. Data ingestion and preprocessing
2. Legal document parsing
3. Question-answer corpus parsing
4. Document construction with metadata
5. Embedding generation
6. Vector storage using Chroma
7. Semantic retrieval
8. Answer generation using Gemini
9. Evaluation using automatic metrics
10. Gradio-based user interface

The overall workflow is:

```text
Input Legal Files
        |
Text Cleaning and Parsing
        |
Document Construction + Metadata
        |
Gemini Embeddings
        |
Chroma Vector Database
        |
Semantic Retrieval
        |
Retrieved Context
        |
Gemini Answer Generation
        |
Generated Legal Answer
```

---

## 4. Retrieval Sources

This project uses two retrieval sources.

### 4.1 Legal Content Corpus

The legal content corpus contains legal provisions extracted from the original legal document. Each entry is structured as:

```text
Mục
Tiêu đề
Nội dung
```

This source preserves the original structure and wording of the legal document. It is useful for answering out-of-set or open-ended legal questions.

### 4.2 Question-Answer Corpus

The QA corpus consists of manually created question-answer pairs based on the same legal document. Each entry is structured as:

```text
Câu x:
Trả lời:
```

This source is useful for FAQ-style questions, especially when the user query is identical or semantically similar to a prepared question.

---

## 5. Technologies Used

The project uses the following main libraries and tools:

```text
Python
LangChain
LangChain Chroma
LangChain Google GenAI
ChromaDB
Gemini Embeddings
Gemini Flash
BERTScore
Pandas
Gradio
Matplotlib
Google Colab
```

---

## 6. Installation

The notebook is designed to run in Google Colab. Install the required libraries by running:

```python
!pip -q install \
  langchain==0.2.16 \
  langchain-core==0.2.38 \
  langchain-community==0.2.16 \
  langchain-google-genai==1.0.8 \
  langchain-text-splitters==0.2.4 \
  langchain-chroma==0.1.4 \
  chromadb==0.5.9 \
  gradio bert-score openpyxl
```

---

## 7. Environment Variables

The project requires a Gemini API key.

Set the API key in the environment:

```python
import os

os.environ["GEMINI_API_KEY"] = "your_api_key_here"
os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
```

For security reasons, API keys should not be hard-coded in public repositories. Use environment variables or a `.env` file instead.

---

## 8. Input File Format

### 8.1 Legal Content File

The legal content file should follow this format:

```text
Mục 1
Tiêu đề: Điều 1 01/2014/NQLT/CP-UBTƯMTTQVN
Nội dung: ...

Mục 2
Tiêu đề: Điều 2 01/2014/NQLT/CP-UBTƯMTTQVN
Nội dung: ...
```

Each legal provision is parsed into a structured record containing:

```text
item_id
title
content
article_number
legal_code
```

### 8.2 QA File

The QA file should follow this format:

```text
Câu 1:
Nghị quyết liên tịch 01/2014/NQLT/CP-UBTƯMTTQVN điều chỉnh những nội dung gì?

Trả lời:
Nghị quyết liên tịch này hướng dẫn phối hợp thực hiện một số quy định của pháp luật về hòa giải ở cơ sở.
```

Each QA entry is parsed into:

```text
qa_id
question
gold_answer
```

---

## 9. Main Pipeline

### Step 1: Text Cleaning

The input text is normalized by removing unnecessary whitespace, invisible Unicode characters, inconsistent line breaks, and blank lines.

### Step 2: Parsing

The legal content file and QA file are parsed using regular expressions. The parsed data are stored in Pandas DataFrames.

### Step 3: Document Construction

The parsed records are converted into LangChain `Document` objects.

For content documents:

```python
page_content = "Tiêu đề: ...\nNội dung: ..."
```

For QA documents:

```python
page_content = "Câu hỏi: ...\nTrả lời: ..."
```

Each document also contains metadata for traceability and evaluation.

### Step 4: Embedding and Indexing

The documents are embedded using Gemini embeddings and stored in two separate Chroma collections:

```text
content_collection
qa_collection
```

### Step 5: Retrieval

The system retrieves relevant documents using semantic similarity search. The project also defines MMR-based retrieval for diversity-aware retrieval.

### Step 6: Answer Generation

The retrieved context is passed to Gemini Flash through a constrained prompt. The model is instructed to answer only from the retrieved context and not to use external knowledge.

### Step 7: Evaluation

The generated answers are evaluated using multiple metrics, including Exact Match, Soft Contains, Token-level F1, BERTScore, Retrieval Hit, Acceptable Match, and latency measurements.

---

## 10. Compared Modes

The main experimental comparison focuses on the following modes:

```text
CONTENT_SIM + FLASH
QA_SIM + FLASH
```

### CONTENT_SIM + FLASH

This mode retrieves information from the original legal content corpus using similarity search and generates answers with Gemini Flash.

It is more suitable for open-ended or out-of-set legal questions.

### QA_SIM + FLASH

This mode retrieves information from the QA corpus using similarity search and generates answers with Gemini Flash.

It is more suitable for FAQ-style questions that are identical or semantically similar to the prepared QA pairs.

---

## 11. Adaptive Hybrid Legal RAG

In addition to the basic retrieval modes, the notebook also introduces an Adaptive Hybrid Legal RAG approach.

This enhanced pipeline includes:

* Query analysis and routing
* Multi-query expansion
* Retrieval from both QA and content sources
* Metadata-aware reranking
* Diversity-aware candidate selection
* Citation-aware prompting
* Grounding validation
* Caching for repeated queries

The hybrid mode is designed to combine the strengths of both QA-based retrieval and content-based retrieval.

---

## 12. Evaluation Metrics

The system is evaluated using the following metrics:

### Exact Match

Checks whether the generated answer exactly matches the reference answer after normalization.

### Soft Contains

Checks whether the generated answer contains the reference answer or vice versa.

### Token-level F1

Measures word-level overlap between the generated answer and the gold answer.

### BERTScore

Measures semantic similarity between the generated answer and the reference answer.

### Retrieval Hit

Checks whether the retrieved context contains sufficient information to answer the question.

### Acceptable Match

A relaxed metric that determines whether the generated answer is practically acceptable based on Soft Contains, Token-level F1, and BERTScore.

### Latency Metrics

The system records:

```text
retrieval_time_sec
generation_time_sec
total_time_sec
```

---

## 13. Evaluation Scenarios

The project evaluates the chatbot on three types of questions:

### 13.1 Exact Questions

Questions that are already present in the QA corpus.

### 13.2 Similar Questions

Paraphrased questions that have the same meaning as the original QA questions.

### 13.3 Out-of-Set Related Questions

Questions that are not included in the QA corpus but are still related to the legal document.

---

## 14. Key Findings

The experimental results show that retrieval source selection has a significant impact on answer quality.

The QA corpus performs better for exact and semantically similar questions because it contains direct question-answer pairs. This makes it effective for FAQ-style legal question answering.

The legal content corpus performs better for out-of-set but related questions because it preserves the original legal provisions and provides broader coverage.

Therefore, a hybrid retrieval architecture is more suitable for practical legal chatbot systems because it can combine the directness of QA retrieval with the broader coverage of content-based retrieval.

---

## 15. Gradio Demo

The project includes Gradio interfaces for testing and comparing retrieval modes.

The basic comparison interface allows users to send the same legal question to both the content source and the QA source. It displays:

```text
Content source answer
Content source context
Content source metadata
Content source latency
QA source answer
QA source context
QA source metadata
QA source latency
```

The Adaptive Hybrid Legal RAG demo displays:

```text
Generated Answer
Routing Decision
Expanded Queries
Retrieved Context
Grounding Validation
Timing
```

To launch the Gradio demo:

```python
demo.launch(share=True)
```

or:

```python
hybrid_demo.launch(share=True)
```

---

## 16. Example Questions

Example legal questions include:

```text
Nghị quyết liên tịch 01/2014/NQLT/CP-UBTƯMTTQVN điều chỉnh những nội dung gì?

Việc phối hợp hoạt động trong công tác hòa giải ở cơ sở được thực hiện trên cơ sở nào?

Ai chịu trách nhiệm kiểm tra việc thực hiện pháp luật về hòa giải ở cơ sở?

Khi nào Nghị quyết này bắt đầu có hiệu lực?

Trong bao lâu trước ngày dự kiến bầu hòa giải viên thì phải chuẩn bị việc bầu?
```

---

## 17. Project Structure

A suggested repository structure is:

```text
vietnamese-legal-rag-chatbot/
│
├── Thesis.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── legal_content.txt
│   └── legal_qa.txt
├── results/
│   ├── exact_results.xlsx
│   ├── similar_results.xlsx
│   └── ood_results.xlsx
└── figures/
    └── evaluation_charts/
```

---

## 18. Limitations

This project has several limitations.

First, the current dataset is small and mainly based on one Vietnamese legal document. Therefore, the results should be interpreted as preliminary findings rather than a large-scale benchmark.

Second, the QA corpus is manually created, so its quality depends on the construction process.

Third, the evaluation set is limited in size, especially for out-of-set related questions.

Fourth, the system does not yet include a complete legal citation validation mechanism.

Fifth, response latency is still high because answer generation by the LLM takes significantly longer than vector retrieval.

---

## 19. Future Work

Future work may include:

* Expanding the corpus to include more Vietnamese legal documents
* Adding more legal domains such as civil law, labor law, administrative law, and education law
* Building a larger and more diverse QA benchmark
* Implementing hybrid lexical-semantic retrieval
* Comparing the system with BM25 and standalone LLM baselines
* Adding reranking and query expansion
* Improving citation-based answer generation
* Adding legal validity checking
* Conducting human evaluation with legal experts
* Reducing latency through caching, context compression, and lighter generation models

---

## 20. Conclusion

This project demonstrates that a Retrieval-Augmented Generation framework can be applied to Vietnamese legal question answering. By combining semantic retrieval with LLM-based generation, the chatbot can produce answers grounded in retrieved legal context.

The results suggest that QA-based retrieval is more effective for exact and similar questions, while content-based retrieval is more robust for out-of-set legal questions. These findings support the development of hybrid retrieval systems for future Vietnamese legal chatbot applications.
