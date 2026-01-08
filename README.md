````md
# 365 Q&A Chatbot

A PDF-based Question & Answer chatbot built with **LangChain**, **Chroma**, **Streamlit**, and **OpenAI**.  
Users can ask questions and receive answers strictly grounded in the uploaded PDF content.

## Setup
```bash
git clone <repo>
cd question_answer_agent
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
````

Create `.env`:

```
OPENAI_API_KEY=your_key
```

## Run

```bash
streamlit run app.py
```
