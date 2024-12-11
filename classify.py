import anthropic
import pandas as pd
import json
import os
from tqdm import tqdm
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import shutil
from datasets import load_dataset
import pickle
import argparse
import numpy as np
import random

# Set fixed seed
SEED = 309


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classify Yelp reviews using Claude API')
    parser.add_argument('--api-key',
                        required=True,
                        help='Anthropic API key')
    parser.add_argument('--num-samples',
                        type=int,
                        default=100,
                        help='Number of reviews to classify (default: 100)')
    parser.add_argument('--output-dir',
                        type=str,
                        default='classified_reviews',
                        help='Directory to save classified reviews (default: classified_reviews)')
    parser.add_argument('--cache-dir',
                        type=str,
                        default='dataset_cache',
                        help='Directory to cache dataset (default: dataset_cache)')
    parser.add_argument('--force-new-sample',
                        action='store_true',
                        help='Force new sample selection even if cached sample exists')
    return parser.parse_args()


class YelpDatasetProcessor:
    def __init__(self, cache_dir: str = "dataset_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.dataset_cache_path = self.cache_dir / "yelp_dataset_cache.pkl"
        self.sample_cache_path = self.cache_dir / f"sample_cache_{SEED}.pkl"

    def load_or_download_dataset(self) -> pd.DataFrame:
        """
        Load dataset from cache if available, otherwise download from Hugging Face
        """
        if self.dataset_cache_path.exists():
            print("Loading dataset from cache...")
            with open(self.dataset_cache_path, 'rb') as f:
                return pickle.load(f)

        print("Downloading dataset from Hugging Face...")
        # Set seed for dataset loading
        np.random.seed(SEED)
        random.seed(SEED)

        dataset = load_dataset("yelp_review_full", split="train")

        df = pd.DataFrame({
            'text': dataset['text'],
            'stars': dataset['label']
        })

        df['review_id'] = [f'review_{i}' for i in range(len(df))]

        with open(self.dataset_cache_path, 'wb') as f:
            pickle.dump(df, f)

        return df

    def prepare_sample_dataset(self, num_samples: int, force_new: bool = False) -> pd.DataFrame:
        """
        Create a random sample from the dataset with fixed seed
        """
        sample_cache_file = self.sample_cache_path.with_name(
            f"sample_cache_{num_samples}_{SEED}.pkl")

        # Return cached sample if exists and not forcing new
        if sample_cache_file.exists() and not force_new:
            print("Loading cached sample...")
            with open(sample_cache_file, 'rb') as f:
                return pickle.load(f)

        print(f"Preparing {num_samples} samples from the dataset...")

        # Set seed for reproducibility
        np.random.seed(SEED)
        random.seed(SEED)

        df = self.load_or_download_dataset()
        sample_df = df.sample(n=min(num_samples, len(df)), random_state=SEED)

        # Cache the sample
        with open(sample_cache_file, 'wb') as f:
            pickle.dump(sample_df, f)

        return sample_df[['review_id', 'text', 'stars']]


class YelpReviewClassifier:
    def __init__(self, api_key: str, output_dir: str = "classified_reviews"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.results_cache_dir = Path("results_cache")
        self.results_cache_dir.mkdir(exist_ok=True)

        # Define valid categories with descriptions
        self.categories = {
            'restaurant': 'Food service establishments, including restaurants, cafes, diners, and food trucks',
            'shopping': 'Retail stores, markets, malls, and other shopping venues',
            'drinks': 'Bars, coffee shops, bubble tea places, and beverage-focused establishments',
            'medical': 'Healthcare facilities, hospitals, clinics, dental offices, and medical services',
            'beauty': 'Hair salons, nail salons, spas, and other beauty service providers',
            'housing': 'Apartments, real estate agencies, property management, and housing-related services',
            'entertainment': 'Movie theaters, museums, game centers, and recreational facilities',
            'others': 'Services and businesses that don\'t fit in the above categories'
        }

        # Example conversations with clear category indicators
        self.example_conversations = [
            {"role": "user", "content": "Review: The Reuben sandwich here was amazing! Huge portions for only $7.50. Staff was friendly and kept our drinks filled. Cash only, but there's an ATM available."},
            {"role": "assistant", "content": "restaurant"},
            {"role": "user", "content": "Review: Best Giant Eagle around! They have beer, huge produce selection, natural foods section, and ethnic food aisles. Great bakery and drive-through pharmacy too."},
            {"role": "assistant", "content": "shopping"},
            {"role": "user", "content": "Review: Coffee is way too strong here. The breakfast burrito was even worse. Complete waste of money at this airport cafe."},
            {"role": "assistant", "content": "drinks"},
            {"role": "user", "content": "Review: New hospital with nice staff but very disorganized. Had to explain symptoms to 8 different people. Simple visit took over 2 hours. Won't return."},
            {"role": "assistant", "content": "medical"},
            {"role": "user", "content": "Review: Got my wedding updo here for $25! Amazing work, took time to get it exactly right. Great value for the quality."},
            {"role": "assistant", "content": "beauty"},
            {"role": "user", "content": "Review: First apartment had roach problems. Management fixed it but too many maintenance visits. They make you decide renewal 6 months early."},
            {"role": "assistant", "content": "housing"},
            {"role": "user", "content": "Review: Great science museum for kids with interactive exhibits on rainforests, ecosystems, and architecture. Less engaging for adults."},
            {"role": "assistant", "content": "entertainment"},
            {"role": "user", "content": "Review: Matt is great with car repairs! Fast, fair pricing, and clearly explains problems before proceeding. Best mechanic in town."},
            {"role": "assistant", "content": "others"}
        ]

    def get_cache_path(self, review_id: str) -> Path:
        """Get the cache path for a specific review"""
        return self.results_cache_dir / f"{review_id}.txt"

    def setup_output_directories(self):
        """Create directories for each category"""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(exist_ok=True)
        for category in self.categories:
            (self.output_dir / category).mkdir(exist_ok=True)

    def classify_single_review(self, review: str, review_id: str) -> str:
        """Classify a single review using Claude API with caching"""
        cache_path = self.get_cache_path(review_id)

        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return f.read().strip()

        try:
            # Construct a more detailed system prompt
            system_prompt = f"""You are a specialized review classifier for business categories. Your task is to analyze business reviews and assign them to the most appropriate category.

Categories and their descriptions:
{chr(10).join(f'- {cat}: {desc}' for cat, desc in self.categories.items())}

Instructions:
1. Read the review carefully and identify key business indicators
2. Match the business type to the most appropriate category
3. If multiple categories could apply, choose the most prominent one
4. If no category clearly fits, use 'others'

Rules:
- Output EXACTLY ONE category name from the list above
- Use lowercase in your response
- Do not include any additional text or explanation
- Do not create new categories

Example format:
User: "Review: [review text]"
Assistant: "[category]"
"""

            # Prepare messages with 'Review:' prefix
            messages = []
            for i in range(0, len(self.example_conversations), 2):
                messages.append({
                    "role": "user",
                    "content": self.example_conversations[i]["content"]
                })
                messages.append({
                    "role": "assistant",
                    "content": self.example_conversations[i + 1]["content"]
                })

            # Add the new review with consistent formatting
            messages.append({
                "role": "user",
                "content": f"Review: {review}"
            })

            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                system=system_prompt,
                messages=messages
            )

            category = response.content[0].text.strip().lower()

            # Validate category
            if category not in self.categories:
                logging.warning(
                    f"Invalid category '{category}' returned. Defaulting to 'others'")
                category = 'others'

            with open(cache_path, 'w') as f:
                f.write(category)

            return category

        except Exception as e:
            logging.error(f"Error classifying review: {str(e)}")
            return "others"

    def save_review_to_file(self, review_text: str, category: str, review_id: str):
        """Save the review to appropriate category folder"""
        try:
            file_path = self.output_dir / category / f"{review_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(review_text)
        except Exception as e:
            logging.error(f"Error saving review {review_id}: {str(e)}")

    def process_reviews(self, df: pd.DataFrame):
        """Process and classify reviews"""
        self.setup_output_directories()

        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying reviews"):
            cache_path = self.get_cache_path(row['review_id'])
            category = self.classify_single_review(
                row['text'], row['review_id'])
            self.save_review_to_file(row['text'], category, row['review_id'])
            results.append({
                'review_id': row['review_id'],
                'category': category,
                'text': row['text'],
                'stars': row['stars']
            })
            if not cache_path.exists():  # avoid frequently requests if we use api
                time.sleep(0.1)

        return pd.DataFrame(results)


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Set all random seeds
    np.random.seed(SEED)
    random.seed(SEED)

    # Initialize processor and classifier
    processor = YelpDatasetProcessor(cache_dir=args.cache_dir)
    classifier = YelpReviewClassifier(
        api_key=args.api_key,
        output_dir=args.output_dir
    )

    try:
        # Prepare sample dataset
        print(f"Preparing {args.num_samples} samples from the dataset...")
        sample_df = processor.prepare_sample_dataset(
            args.num_samples,
            force_new=args.force_new_sample
        )

        # Process and classify reviews
        print("Starting classification...")
        results_df = classifier.process_reviews(sample_df)

        # Save results summary
        results_filename = "classification_results.csv"
        results_df.to_csv(results_filename, index=False)

        # Print statistics
        print("\nClassification Results:")
        print(results_df['category'].value_counts())

        print("\nAverage Stars per Category:")
        print(results_df.groupby('category')['stars'].mean().round(2))

        print("\nCategory Distribution (%):")
        category_dist = (
            results_df['category'].value_counts() / len(results_df) * 100).round(2)
        print(category_dist)

        print(
            f"\nReviews have been saved in category-specific folders under '{args.output_dir}/'")
        print(f"Results saved to {results_filename}")

    except Exception as e:
        print(f"Error during classification: {str(e)}")


if __name__ == "__main__":
    main()
