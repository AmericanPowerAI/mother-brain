import os, zstd, json, re

class Omni:
    def __init__(self):
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self):
        if os.path.exists('knowledge.zst'):
            return json.loads(zstd.decompress(open('knowledge.zst','rb').read()))
        return {"_meta": {"version": 0}}
    
    def _save_knowledge(self):
        open('knowledge.zst','wb').write(zstd.compress(json.dumps(self.knowledge).encode()))
    
    def learn(self, text):
        facts = re.findall(r'(?:[A-Z][a-z]+\s?){2,5}(?=\s|\.)|(?<!\d)\d{4}(?!\d)|[A-Z]{3,}', text)
        for fact in facts:
            self.knowledge[fact[:128]] = text[:1024]
        self._save_knowledge()
    
    def answer(self, query):
        return next((v for k,v in self.knowledge.items() if k in query), "I don't know")

if __name__ == "__main__":
    omni = Omni()
    print("Omni Brain v0.1")
    while True:
        omni.learn(input("Teach me > "))
        print(omni.answer(input("Ask me > ")))
