import json, lzma, re

class Mother:
    def __init__(self):
        try:
            with lzma.open('knowledge.zst', 'rb') as f:
                self.knowledge = json.load(f)
        except:
            self.knowledge = {"_meta": {"name": "mother-brain"}}
    
    def learn(self, text):
        facts = re.findall(r'\b[A-Z]\w+(?:\s+[A-Z]\w+)+', text)  # Better pattern
        for fact in set(facts):  # Deduplicate
            self.knowledge[fact] = text[:500]  # Store snippet
        self._save()
    
    def _save(self):
        with lzma.open('knowledge.zst', 'wb') as f:
            json.dump(self.knowledge, f)
    
    def answer(self, query):
        query = query.lower()
        return next((v for k,v in self.knowledge.items() 
                   if query in k.lower()), "Learning mode active")

if __name__ == "__main__":
    m = Mother()
    while True:
        m.learn(input("Teach me > "))
        print(m.answer(input("Ask me > ")))
