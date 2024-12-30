import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class HybridRecommender:
    def __init__(self):
        """
        Initialize the HybridRecommender class.
        """
        self.collaborative_model = None
        self.project_features = None
        self.user_preferences = None
        self.max_donation = None

    def load_data(self, interactions_file, projects_file, users_file):
        """
        Load data from CSV files containing user interactions, projects, and user information.

        Args:
            interactions_file (str): Path to CSV file containing user-project interactions
            projects_file (str): Path to CSV file containing project details
            users_file (str): Path to CSV file containing user information
        """
        self.interactions = pd.read_csv(interactions_file)
        self.projects = pd.read_csv(projects_file)
        self.users = pd.read_csv(users_file)

    def preprocess_data(self):
        """
        Preprocess interaction data by normalizing donation amounts to a 0-1 scale.

        Returns:
            surprise.dataset.Dataset: Processed dataset ready for collaborative filtering
        """
        # Normalize donation amounts to a 0-1 scale
        self.max_donation = self.interactions['donation_amount'].max()
        interactions_normalized = self.interactions.copy()
        interactions_normalized['donation_amount'] = (
            self.interactions['donation_amount'] / self.max_donation
        )

        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(
            interactions_normalized[["user_id", "project_id", "donation_amount"]], reader
        )
        return data

    def plot_donation_distribution(self):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.interactions['donation_amount'], bins=30, kde=True, color='blue')
        plt.title('Distribution of Donation Amounts', fontsize=16)
        plt.xlabel('Donation Amount', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.show()
    
    def plot_total_donations_by_category(self):
        merged = self.interactions.merge(self.projects, on="project_id")
        category_donations = merged.groupby("category")["donation_amount"].sum().sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        category_donations.plot(kind="bar", color="skyblue")
        plt.title("Total Donations by Project Category", fontsize=16)
        plt.xlabel("Category", fontsize=14)
        plt.ylabel("Total Donations", fontsize=14)
        plt.xticks(rotation=45)
        plt.show()

    def plot_top_users_by_donations(self):
        user_donations = self.interactions.groupby("user_id")["donation_amount"].count().nlargest(10)

        plt.figure(figsize=(12, 6))
        user_donations.plot(kind="bar", color="purple")
        plt.title("Top 10 Users by Number of Donations", fontsize=16)
        plt.xlabel("User ID", fontsize=14)
        plt.ylabel("Number of Donations", fontsize=14)
        plt.show()

    def train_collaborative_model(self, data):
        """
        Train the collaborative filtering model using SVD algorithm.

        Args:
            data (surprise.dataset.Dataset): Preprocessed dataset for training
        """
        trainset, _ = surprise_train_test_split(data, test_size=0.2, random_state=42)
        self.collaborative_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        self.collaborative_model.fit(trainset)

    def train_content_based_model(self):
        """
        Train the content-based model by creating project feature dictionaries
        and user preference mappings based on project categories.
        """
        self.project_features = self.projects.set_index("project_id")["category"].to_dict()
        self.user_preferences = self.interactions.groupby("user_id")["project_id"].apply(list).to_dict()

    def knowledge_based_recommendation(self, user_id):
        """
        Generate knowledge-based recommendations by matching user interests with project categories.

        Args:
            user_id: The ID of the user to generate recommendations for

        Returns:
            list: List of recommended project IDs based on user interests
        """
        user_info = self.users[self.users["user_id"] == user_id].iloc[0]
        user_interests = user_info["interests"]

        recommended_projects = self.projects[self.projects["category"] == user_interests]
        return recommended_projects["project_id"].tolist()


    def hybrid_recommendation(self, user_id):
        """
        Generate hybrid recommendations by combining collaborative filtering and content-based approaches.

        Args:
            user_id: The ID of the user to generate recommendations for

        Returns:
            list: Top 10 recommended project IDs based on hybrid scoring

        Raises:
            ValueError: If models haven't been trained before making recommendations
        """
        if not self.collaborative_model or not self.project_features:
            raise ValueError("Models have not been trained. Please train them before making recommendations.")

        # Get Collaborative Filtering recommendations
        unique_projects = self.interactions["project_id"].unique()
        cf_scores = {
            project_id: self.collaborative_model.predict(user_id, project_id).est * self.max_donation
            for project_id in unique_projects
        }
        user_projects = self.user_preferences.get(user_id, [])
        content_scores = {
            project_id: sum(
                self.project_features[project_id] == self.project_features.get(p, "")
                for p in user_projects
            )
            for project_id in unique_projects
        }
        hybrid_scores = {
            project_id: cf_scores.get(project_id, 0) + content_scores.get(project_id, 0)
            for project_id in unique_projects
        }
        # Sort recommendations
        recommended_projects = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        return [project_id for project_id, _ in recommended_projects]
