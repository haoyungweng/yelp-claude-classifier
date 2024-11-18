# Yelp Review Classifier

This script classifies Yelp reviews into categories using the Claude AI API.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yelp-review-classifier.git
cd yelp-review-classifier
```

2. Create a conda environment and install requirements:
```bash
conda create -n yelp_classifier python=3.10
conda activate yelp_classifier
pip install -r requirements.txt
```

## Usage

Run the script with your Anthropic API key:
```bash
python yelp_classifier.py --api-key "your-api-key-here"
```

Optional arguments:
- `--num-samples`: Number of reviews to classify (default: 100)
- `--output-dir`: Directory to save classified reviews (default: classified_reviews)
- `--cache-dir`: Directory to cache dataset (default: dataset_cache)

Example:
```bash
python yelp_classifier.py --api-key "your-api-key-here" --num-samples 200
```

## Categories

Reviews are classified into the following categories:
- restaurant
- shopping
- drinks
- medical
- beauty
- housing
- entertainment
- others

## Output

The script creates:
1. Separate folders for each category containing the classified reviews
2. A CSV file with classification results and statistics