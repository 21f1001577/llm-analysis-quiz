# 📘 **LLM Analysis Quiz – Automated Quiz Solver**

This project is a fully automated quiz-analysis system powered by LLMs and browser automation.  
It extracts quiz data, processes it, sends it through an LLM, and returns structured answers — all through a simple API.

## 🚀 **Features**
- Browser automation via Selenium to fetch quiz content  
- Structured data extraction and cleaning  
- LLM-powered reasoning  
- Configurable model selection  
- Logging and temp handling  
- API backend for triggering quiz solving  

## 📂 **Project Structure**
```
app.py
browser_handler.py
config.py
data_processor.py
llm_helper.py
quiz_solver.py
downloads/
temp/
logs/
.env
LICENSE
pyproject.toml
```



## 🔐 Environment Variables
```
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
MODEL_NAME=your_default_model
BROWSER_HEADLESS=true
DOWNLOAD_DIR=downloads
```

## ▶️ Running
```
uvicorn app:app --reload
```

## 📦 Deployment
Heroku, Render, Docker supported.

## 📄 License
MIT License (see LICENSE file)
