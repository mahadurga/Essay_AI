import os
import logging
import argparse
import kagglehub
import pandas as pd
from modules.model_trainer import EssayFeedbackModel, prepare_ielts_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_ielts_dataset():
    """
    Download the IELTS Writing Scored Essays dataset from Kaggle
    
    Returns:
        str: Path to the downloaded dataset
    """
    try:
        logger.info("Downloading IELTS Writing Scored Essays dataset...")
        path = kagglehub.dataset_download("mazlumi/ielts-writing-scored-essays-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def find_csv_in_directory(directory):
    """
    Find the first CSV file in a directory
    
    Args:
        directory (str): Directory to search
        
    Returns:
        str: Path to the CSV file, or None if not found
    """
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            return os.path.join(directory, file)
    return None

def main():
    parser = argparse.ArgumentParser(description='Train essay feedback models using the IELTS dataset')
    parser.add_argument('--download', action='store_true', help='Download the dataset from Kaggle')
    parser.add_argument('--dataset-path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save trained models')
    args = parser.parse_args()
    
    try:
        # Get dataset path
        dataset_path = args.dataset_path
        if args.download or dataset_path is None:
            download_path = download_ielts_dataset()
            if os.path.isdir(download_path):
                dataset_path = find_csv_in_directory(download_path)
            else:
                dataset_path = download_path
        
        if dataset_path is None:
            raise ValueError("Could not find dataset CSV file")
        
        logger.info(f"Using dataset: {dataset_path}")
        
        # Prepare dataset
        prepared_data_dir = os.path.join(os.path.dirname(dataset_path), 'prepared')
        os.makedirs(prepared_data_dir, exist_ok=True)
        prepared_data_path = os.path.join(prepared_data_dir, 'prepared_ielts_dataset.csv')
        
        prepare_ielts_dataset(dataset_path, prepared_data_path)
        
        # Initialize and train models
        model_trainer = EssayFeedbackModel(model_dir=args.model_dir)
        models = model_trainer.train_all_models(prepared_data_path)
        
        logger.info("Models trained and saved successfully")
        
        # Test prediction
        df = pd.read_csv(dataset_path)
        if len(df) > 0:
            test_essay = df.iloc[0]['Essay']
            predictions = model_trainer.predict(test_essay)
            logger.info(f"Test predictions for first essay: {predictions}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())