import json, lzma, re

class Mother:
    def __init__(self):
        try:
            self.knowledge = json.loads(lzma.decompress(open('knowledge.zst','rb').read())
        except:
            self.knowledge = {"_meta": {"name": "mother-brain"}}
    
    def learn(self, text):
        facts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b|\d{4}', text)  # Simple pattern
        for fact in facts[:5]:  # Limit to 5 facts per learning
            self.knowledge[fact] = text[:500]  # Trim long texts
        self._save()
    
    def _save(self):
        open('knowledge.zst','wb').write(lzma.compress(json.dumps(self.knowledge).encode()))
    
    def answer(self, query):
        return next((v for k,v in self.knowledge.items() if k.lower() in query.lower()), 
                   "I'm still learning")

if __name__ == "__main__":
    m = Mother()
    print("Mother Brain v0.1 (Ctrl+C to exit)")
    while True:
        m.learn(input("Teach me > "))
        print(m.answer(input("Ask me > ")))
