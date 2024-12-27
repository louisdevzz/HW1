import random
import pandas as pd
from faker import Faker

# Initialize Faker for generating random names and descriptions
fake = Faker()

# Define constants
NUM_USERS = 1000
NUM_PROJECTS = 1000
NUM_INTERACTIONS = 10000

# Generate users
def generate_users(num_users):
    users = []
    for user_id in range(1, num_users + 1):
        user = {
            "user_id": user_id,
            "name": fake.name(),
            "email": fake.email(),
            "interests": random.choice(["Education", "Health", "Environment", "Technology", "Community"]),
        }
        users.append(user)
    return pd.DataFrame(users)

# Generate projects
def generate_projects(num_projects):
    projects = []
    for project_id in range(1, num_projects + 1):
        project = {
            "project_id": project_id,
            "title": fake.sentence(nb_words=4),
            "description": fake.paragraph(nb_sentences=3),
            "category": random.choice(["Education", "Health", "Environment", "Technology", "Community"]),
            "impact_score": round(random.uniform(1, 5), 2),
            "location": fake.city(),
        }
        projects.append(project)
    return pd.DataFrame(projects)

# Generate interactions
def generate_interactions(num_interactions, user_ids, project_ids):
    interactions = []
    
    # First, ensure each user has at least one interaction
    for user_id in user_ids:
        interaction_type = random.choice(["view", "like", "donate"])
        interaction = {
            "user_id": user_id,
            "project_id": random.choice(project_ids),
            "donation_amount": round(random.uniform(10, 500), 2) if interaction_type == "donate" else 0,
            "interaction_type": interaction_type,
        }
        interactions.append(interaction)
    
    # Generate remaining random interactions
    remaining_interactions = num_interactions - len(user_ids)
    if remaining_interactions > 0:
        for _ in range(remaining_interactions):
            interaction_type = random.choice(["view", "like", "donate"])
            interaction = {
                "user_id": random.choice(user_ids),
                "project_id": random.choice(project_ids),
                "donation_amount": round(random.uniform(10, 500), 2) if interaction_type == "donate" else 0,
                "interaction_type": interaction_type,
            }
            interactions.append(interaction)
    
    return pd.DataFrame(interactions)

# Main script
if __name__ == "__main__":
    # Generate data
    users_df = generate_users(NUM_USERS)
    projects_df = generate_projects(NUM_PROJECTS)
    interactions_df = generate_interactions(NUM_INTERACTIONS, users_df["user_id"].tolist(), projects_df["project_id"].tolist())

    # Save to CSV
    users_df.to_csv("users.csv", index=False)
    projects_df.to_csv("projects.csv", index=False)
    interactions_df.to_csv("interactions.csv", index=False)

    print("Sample data generated:")
    print(f"- Users: {len(users_df)} records")
    print(f"- Projects: {len(projects_df)} records")
    print(f"- Interactions: {len(interactions_df)} records")
