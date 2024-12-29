import HybridRecommender

# Initialize the recommender
recommender = HybridRecommender()

# Load the data
recommender.load_data("interactions.csv", "projects.csv", "users.csv")

# Preprocess the data
data = recommender.preprocess_data()

# Print the first 5 ratings from the trainset
trainset = data.build_full_trainset()

for idx, (uid, iid, rating) in enumerate(trainset.all_ratings()):
    if idx >= 5:
        break
    print(f"User ID: {trainset.to_raw_uid(uid)}, Project ID: {trainset.to_raw_iid(iid)}, Donation (normalized): {rating}")


# Train the collaborative model
recommender.train_collaborative_model(data)

# Train the content-based model
recommender.train_content_based_model()

# Generate knowledge-based recommendations
recommender.knowledge_based_recommendation(user_id=123)

# Generate hybrid recommendations
recommender.hybrid_recommendation(user_id=123)