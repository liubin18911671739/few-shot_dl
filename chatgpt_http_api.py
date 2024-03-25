import requests
import json
import sqlite3


class ChatGPT:
    def __init__(self, api_key, proxies=None):
        self.api_key = api_key
        self.proxies = proxies
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.url = "https://api.openai.com/v1/chat/completions"

    def ask(self, prompt):
        data = {
            "model": "gpt-3.5-turbo",  # 根据需要选择合适的模型
            "messages": [{"role": "user", "content": prompt}],
        }
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(data), proxies=self.proxies
        )
        return response.json()  # 返回JSON解析后的响应


# 替换为您的API密钥
api_key = "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"

GPT3 = ChatGPT(api_key)

# 发送问题并获取回答
prompt = "list the 5 most recent research directions in the field of Artificial Intelligence, research content, important concepts, important scholars, important papers, important publications and important research institutions."
prompt1 = "列出人工智能领域的 5 个最新研究方向、研究内容、重要概念、重要学者、重要论文、重要出版物和重要研究机构。"

data = GPT3.ask(prompt)

# print(data)

for choice in data["choices"]:
    if "message" in choice and "content" in choice["message"]:
        print(choice["message"]["content"])
# result["choices"][0]["message"]["content"]
