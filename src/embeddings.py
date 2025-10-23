import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load a small, open-source embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, fast and free


def email_to_vector(email_text: str) -> np.ndarray:
    """Convert a full email (subject + body) into an embedding vector."""
    return model.encode(email_text, normalize_embeddings=True)


def k_nearest_majority_vote(
    email_vec: np.ndarray, spam_vectors: np.ndarray, ham_vectors: np.ndarray, k: int = 5
) -> str:
    """Classify email_vec as spam/ham via cosine-similarity nearest neighbors."""
    all_vectors = np.vstack([spam_vectors, ham_vectors])
    labels = np.array(["spam"] * len(spam_vectors) + ["ham"] * len(ham_vectors))

    sims = cosine_similarity(email_vec.reshape(1, -1), all_vectors).ravel()
    top_k_idx = np.argsort(sims)[-k:]  # indices of k most similar
    top_k_labels = labels[top_k_idx]

    # majority vote
    pred = max(set(top_k_labels), key=list(top_k_labels).count)
    return pred


if __name__ == "__main__":
    spam_examples = [
        "Congratulations! You've been selected for a free gift card!",
        "Claim your lottery prize now—limited time offer!",
        "URGENT: Your account will be suspended unless you verify now!",
        "You've won $1,000,000! Click here to claim your prize!",
        "Your PayPal account has been locked. Confirm your identity immediately.",
        "Lose 30 pounds in 30 days with this one weird trick!",
    ]

    ham_examples = [
        "Hi Sam, here's the agenda for tomorrow's meeting.",
        "Can we reschedule our coffee chat to next week?",
        "Thanks for your help on the project yesterday.",
        "Your order #12345 has been confirmed and will ship tomorrow.",
        "Happy birthday! Hope you have a wonderful day.",
        "Meeting notes from today's call are now available in the shared folder.",
    ]

    spam_vectors = np.vstack([email_to_vector(e) for e in spam_examples])
    ham_vectors = np.vstack([email_to_vector(e) for e in ham_examples])

    new_email = "Reminder: Your Amazon package has shipped."
    email_vec = email_to_vector(new_email)
    label = k_nearest_majority_vote(email_vec, spam_vectors, ham_vectors)
    print(label)
