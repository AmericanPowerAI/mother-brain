# test_feedback.py
from mother import mother

# Test the feedback system
print("Testing UnifiedFeedbackLearner integration...")

# Record a test interaction
mother.feedback_learner.record_interaction(
    question="What is Python?",
    answer="Python is a high-level programming language known for its simplicity and readability.",
    feedback=1  # Positive feedback
)

# Check the statistics
stats = mother.feedback_learner.get_learning_statistics()
print("\nLearning Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Test quality prediction
quality = mother.feedback_learner.predict_answer_quality(
    "What is JavaScript?",
    "JavaScript is a programming language used for web development."
)
print(f"\nPredicted quality for test answer: {quality:.2%}")

print("\n✅ Integration successful!")
