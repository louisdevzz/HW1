import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split

# Load data
def load_data():
    interactions = pd.read_csv("interactions.csv")
    projects = pd.read_csv("projects.csv")
    users = pd.read_csv("users.csv")
    return interactions, projects, users

# Preprocess data for Surprise
def preprocess_data(interactions):
    # Normalize donation amounts to a 0-1 scale
    max_donation = interactions['donation_amount'].max()
    interactions_normalized = interactions.copy()
    interactions_normalized['donation_amount'] = interactions['donation_amount'] / max_donation
    
    reader = Reader(rating_scale=(0, 1))  # Change scale to 0-1
    data = Dataset.load_from_df(interactions_normalized[["user_id", "project_id", "donation_amount"]], reader)
    return data, max_donation

# Train Collaborative Filtering model
def train_collaborative_model(data):
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)  # Tune hyperparameters
    model.fit(trainset)
    return model

# Train Content-Based model (simple similarity)
def train_content_based_model(projects, interactions):
    project_features = projects.set_index("project_id")["category"].to_dict()
    user_preferences = interactions.groupby("user_id")["project_id"].apply(list).to_dict()
    return project_features, user_preferences

# Knowledge-Based Recommendations
def knowledge_based_recommendation(user_id, users, projects):
    # Get user information
    user_info = users[users["user_id"] == user_id].iloc[0]
    user_interests = user_info["interests"]

    # Filter projects by user's interests
    recommended_projects = projects[projects["category"] == user_interests]

    return recommended_projects["project_id"].tolist()

# Hybrid Recommendation System
def hybrid_recommendation(user_id, collaborative_model, content_data, interactions, max_donation):
    project_features, user_preferences = content_data
    
    # Get Collaborative Filtering recommendations
    unique_projects = interactions["project_id"].unique()
    cf_scores = {
        project_id: collaborative_model.predict(user_id, project_id).est * max_donation  # Denormalize predictions
        for project_id in unique_projects
    }

    # Get Content-Based scores
    user_projects = user_preferences.get(user_id, [])
    content_scores = {}
    for project_id in unique_projects:
        content_scores[project_id] = sum(
            project_features[project_id] == project_features.get(p, "")
            for p in user_projects
        )

    # Combine scores
    hybrid_scores = {
        project_id: cf_scores.get(project_id, 0) + content_scores.get(project_id, 0)
        for project_id in unique_projects
    }

    # Sort recommendations
    recommended_projects = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return [project_id for project_id, _ in recommended_projects]

# Main script
if __name__ == "__main__":
    # Step 1: Load data
    interactions_df, projects_df, users_df = load_data()

    # Step 2: Preprocess data
    surprise_data, max_donation = preprocess_data(interactions_df)

    # Step 3: Train Collaborative Filtering model
    collaborative_model = train_collaborative_model(surprise_data)

    # Step 4: Train Content-Based model
    content_data = train_content_based_model(projects_df, interactions_df)

    # Step 5: Test Hybrid Recommendation
    test_user_id = 1  # Example user ID
    hybrid_recommendations = hybrid_recommendation(test_user_id, collaborative_model, content_data, interactions_df, max_donation)
    print(f"Hybrid Recommendations for User {test_user_id}: {hybrid_recommendations}")

    # Step 6: Test Knowledge-Based Recommendation
    knowledge_recommendations = knowledge_based_recommendation(test_user_id, users_df, projects_df)
    print(f"Knowledge-Based Recommendations for User {test_user_id}: {knowledge_recommendations}")
