;import json, lzma, re, os, requests
from flask import Flask, request, jsonify

app = Flask(__name__)

class Mother:
    def __init__(self):
        if not os.path.exists('knowledge.zst'):
            self._init_knowledge()
        self.load()

    def _init_knowledge(self):
        with lzma.open('knowledge.zst', 'wb') as f:
            json.dump({"_meta": {"name": "mother-brain"}}, f)

    def load(self):
        with lzma.open('knowledge.zst', 'rb') as f:
            self.knowledge = json.load(f)

    def learn(self, text):
        facts = re.findall(r'\b[A-Z]\w+(?:\s+[A-Z]\w+)+|\d{4}s?', text)
        for fact in set(facts):
            self.knowledge[fact] = text[:500]
        self._save()

    def _save(self):
        with lzma.open('knowledge.zst', 'wb') as f:
            json.dump(self.knowledge, f)

    def scrape_learn(self, url):
        try:
            text = requests.get(url).text[:2000]  # Limit to first 2000 chars
            self.learn(text)
            return True
        except:
            return False

    def answer(self, query):
        query = query.lower()
        return next((v for k,v in self.knowledge.items() 
                   if query in k.lower()), "I'm still learning")

mother = Mother()

@app.route('/auto-learn', methods=['POST'])
def auto_learn():
    mother.scrape_learn('https://en.wikipedia.org/wiki/Special:Random')
    return jsonify({"status": "learned from random Wikipedia page"})

@app.route('/ask', methods=['GET'])
def ask():
    return jsonify({"answer": mother.answer(request.args.get('q', ''))})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
