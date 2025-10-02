markdown# MOTHER BRAIN - Self-Learning AI System

An autonomous AI system that learns from web data and gradually becomes independent from pre-trained models.

## Features

- **Self-Training Neural Network**: Trains on web-scale datasets to develop independent AI capabilities
- **90-Day Independence Plan**: Gradually phases out dependency on DialoGPT
- **Web-Scale Knowledge**: Learns from Common Crawl, Reddit, GitHub, and more
- **Real-Time Learning**: Feedback system that improves with every interaction
- **Planet-Wide Scanning**: Monitors 14,000+ domains for continuous learning

## Architecture
MOTHER BRAIN
├── ConversationalModel (Current: DialoGPT)
├── MotherTrainer (Training new model)
│   ├── Dataset Processing
│   ├── Vocabulary Building
│   └── LSTM Neural Network
├── Search Engine (Multi-source verification)
├── Feedback Learner (Pattern recognition)
└── Knowledge Database (GitHub-backed)

## Training Progress

The system transitions from DialoGPT to custom model over 90 days:

- **Days 1-30**: Data collection & processing
- **Days 31-60**: Hybrid responses (mixing both models)
- **Days 61-90**: Full independence from DialoGPT

## API Endpoints

- `POST /enhanced-chat` - Chat with AI
- `GET /train/status` - Check training progress
- `POST /train/start` - Begin training process
- `POST /feedback` - Provide feedback for learning
- `POST /search` - Search with verification

## Deployment

Currently deployed on Render: https://mother-brain.onrender.com

### Environment Variables
GITHUB_FINE_GRAINED_PAT=your_github_token

## Dataset Sources

- Common Crawl (Web pages)
- Reddit conversations
- GitHub code repositories
- Wikipedia articles
- News articles
- TinyStories (conversation training)

## Tech Stack

- Python 3.11
- PyTorch 2.8.0
- Transformers (Hugging Face)
- Flask web framework
- GitHub API for persistence

## Development Status

🔄 **Training Phase**: Building independent neural network
- Current: Using DialoGPT fallback
- Goal: Complete independence in 90 days
- Progress: Infrastructure complete, training initiated

## Contributing

This is an experimental self-learning AI system. Contributions welcome!

## License

MIT

---
*Building AI that learns and grows independently*
