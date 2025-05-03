import json, lzma, re, os
from flask import Flask, request, jsonify

app = Flask(__name__)

class Mother:
    def __init__(self):
        # Initialize empty knowledge if file doesn't exist
        if not os.path.exists('knowledge.zst'):
            with lzma.open('knowledge.zst', 'wb') as f:
                json.dump({"_meta": {"name": "mother-brain"}}, f)
        
        with lzma.open('knowledge.zst', 'rb') as f:
            self.knowledge = json.load(f)
    
    def learn(self, text):
        facts = re.findall(r'\b[A-Z]\w+(?:\s+[A-Z]\w+)+|\d{4}s?|\w+ly\b', text)
        for fact in set(facts):
            self.knowledge[fact] = text[:500]
        self._save()
    
    def _save(self):
        with lzma.open('knowledge.zst', 'wb') as f:
            json.dump(self.knowledge, f)
    
    def answer(self, query):
        query = query.lower()
        return next((v for k,v in self.knowledge.items() 
                   if query in k.lower()), "I'm still learning")

mother = Mother()

@app.route('/learn', methods=['POST'])
def learn():
    data = request.json
    mother.learn(data['text'])
    return jsonify({"status": "success"})

@app.route('/ask', methods=['GET'])
def ask():
    query = request.args.get('q', '')
    return jsonify({"answer": mother.answer(query)})

@app.route('/')
def home():
    return """
    <h1>Mother Brain API</h1>
    <p>Endpoints:</p>
    <ul>
        <li>POST /learn - Teach new things (send JSON with 'text')</li>
        <li>GET /ask?q=question - Get answers</li>
    </ul>
    """

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
