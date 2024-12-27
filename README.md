# Donation Recommendation System

## Overview
This project implements a Hybrid Recommendation System for donation suggestions, combining collaborative filtering, content-based filtering, and knowledge-based approaches to provide personalized donation recommendations to users.

## Features

### Hybrid Recommendation Approach
Our system integrates three recommendation strategies:

1. **Collaborative Filtering**
   - Implements SVD (Singular Value Decomposition) algorithm
   - Uses historical donation amounts as ratings
   - Predicts user preferences based on donation patterns

2. **Content-Based Filtering**
   - Analyzes project categories
   - Builds user preference profiles from donation history
   - Matches users with similar project categories

3. **Knowledge-Based Recommendations**
   - Utilizes explicit user interests
   - Matches users with projects in their interest categories
   - Provides personalized recommendations based on user profiles

## System Benefits

- **Multi-Strategy Approach**: Combines three different recommendation methods
- **Scalable Architecture**: Handles multiple data sources efficiently
- **Personalization**: Considers both implicit and explicit user preferences
- **Flexible Rating System**: Handles donation amounts as ratings (0-500 scale)

## Technical Implementation

### Data Processing
- Uses pandas for data manipulation and analysis
- Processes three main data sources:
  - `interactions.csv`: User-project donation history
  - `projects.csv`: Project metadata and categories
  - `users.csv`: User profiles and interests
- Implements Surprise library's Reader for rating scale normalization

### Algorithms

1. **Collaborative Filtering**
   - Uses Surprise's SVD algorithm
   - Implements 80-20 train-test split
   - Evaluates using Mean Squared Error (MSE)
   - Predicts donation amounts for user-project pairs

2. **Content-Based Filtering**
   - Creates project category dictionary
   - Builds user preference profiles from interaction history
   - Calculates similarity based on category matching

3. **Knowledge-Based System**
   - Filters projects based on user interests
   - Direct category matching from user profiles
   - Interest-based project recommendations

4. **Hybrid Integration**
   - Combines collaborative and content-based scores
   - Weighted scoring system for final recommendations
   - Returns top 10 personalized project suggestions

### Key Components
```python
- load_data(): Loads interaction, project, and user data
- preprocess_data(): Prepares data for Surprise library
- train_collaborative_model(): Trains SVD model
- train_content_based_model(): Builds category-based recommendations
- knowledge_based_recommendation(): Generates interest-based suggestions
- hybrid_recommendation(): Combines multiple recommendation approaches
```

### Performance Metrics
- Collaborative Filtering MSE tracking
- Top-10 recommendation generation
- Multiple recommendation sources for comparison

## Getting Started

### Prerequisites
```
Python 3.8+
pandas
numpy
scikit-learn
surprise
```

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare your data files:
   - interactions.csv
   - projects.csv
   - users.csv
4. Run the main application:
```bash
python main.py
```

## Future Improvements
- Enhanced weighting system for hybrid recommendations
- Additional feature engineering for projects
- Real-time recommendation updates
- Advanced evaluation metrics implementation 