# Multi-Agent-With-Langgraph

1. Tài liệu:

- Langgraph Python: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
- LanggraphJS: https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor/

2. Với python:

- Khởi tạo môi trường conda: conda create -n myenv python=3.11
- Active enviroment ở part 1 qua câu lệnh: conda activate myenv
- Cài đặt package qua lệnh: pip install -r requirements.txt
- Chạy file: python multi_agent-Langgraph.py

3. Với Typescript:

- Cài đặt môi trường qua lệnh: npm install
- Chạy file: tsx multi_agent-Langgraph.ts

# Các bước cài đặt thủ công

1. Python

- pip install -U langgraph langchain_community langchain-openai langchain_experimental
- pip install python-dotenv

2. Typescript

- sudo npm install -g tsx
- npm install @langchain/core @langchain/langgraph @langchain/openai @langchain/community dotenv
