import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


# Define an abstract class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a file path.

        """
        pass


# Define a concrete class for ingesting data from a zip file
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """
        Extracts data from a zip file and returns a pandas DataFrame.
        """
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if the file is a zip file
        if not file_path.endswith(".zip"):
            raise ValueError(f"File is not a zip file: {file_path}")

        # Define the extraction path (relative to the script's directory)
        extraction_path = os.path.join(os.path.dirname(file_path), "../../data/raw")
        extraction_path = os.path.abspath(extraction_path)  # Convert to absolute path

        # Create the directory if it doesn't exist
        # os.makedirs(extraction_path, exist_ok=True)

        # Extract the zip file to the specified directory
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extraction_path)

        # Find the extracted CSV file
        extracted_files = os.listdir(extraction_path)
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV file found in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV files found in the extracted data.")

        # Read the CSV file into a DataFrame
        csv_file_path = os.path.join(extraction_path, csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Returns the appropriate data ingestor based on the file extension.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


# Example usage
if __name__ == "__main__":
    # Specify the file path
    file_path = "data\raw\train.csv.zip"
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate data ingestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    # Now df contains the data (extracted CSV) from the zip file
    df.head()
