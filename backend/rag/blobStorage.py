import os
from azure.storage.blob import BlobServiceClient
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Load Azure Blob Storage connection info from environment variables
AZURE_BLOB_ACCOUNT_URL = os.getenv("BLOB_SAS_URL")  # Azure Blob Storage URL with SAS token
CONTAINER_NAME = os.getenv("BLOB_CONTAINER_NAME")  # Blob container name

if not AZURE_BLOB_ACCOUNT_URL or not CONTAINER_NAME:
    raise ValueError("Azure Blob Storage account URL or container name is not set.")

# Create BlobServiceClient once
blob_service_client = BlobServiceClient(account_url=AZURE_BLOB_ACCOUNT_URL)


def upload_blob(file_data: bytes, filename: str) -> bool:
    """
    Upload binary data to Azure Blob Storage container.
    Returns True on success.
    """
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(filename)
        blob_client.upload_blob(file_data, overwrite=True)
        return True
    except Exception as e:
        print(f"Failed to upload blob: {e}")
        return False

def delete_blob(filename: str) -> bool:
    """
    Deletes a blob from the container.
    """
    try:
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(filename)
        # check if exists before deleting to avoid error
        if blob_client.exists():
            blob_client.delete_blob()
            print(f"Blob '{filename}' deleted successfully.")
            return True
        print(f"Blob '{filename}' not found, skipping delete.")
        return True # Return True as the goal (not existing) is met
    except Exception as e:
        print(f"Failed to delete blob: {e}")
        return False

def extract_filename_from_url(blob_url: str) -> str:
    """
    Safely extracts the filename from the full URL based on the
    environment configuration, mirroring the logic in download_blob_to_local.
    """
    prefix = f"{AZURE_BLOB_ACCOUNT_URL}/{CONTAINER_NAME}/"
    if blob_url.startswith(prefix):
        return blob_url[len(prefix):]
    
    # Fallback: if the prefix doesn't match (e.g. env var changed), try splitting
    return blob_url.split("/")[-1]


def download_blob_to_local(blob_url: str) -> str:
    """
    Downloads an Azure Blob to a local temporary file and returns the local file path.

    Args:
        blob_url (str): Full URL of the blob to download.

    Returns:
        str: Path to the downloaded local temp file.
    """
    prefix = f"{AZURE_BLOB_ACCOUNT_URL}/{CONTAINER_NAME}/"
    if not blob_url.startswith(prefix):
        raise ValueError("Blob URL does not match account URL and container name")

    blob_name = blob_url[len(prefix):]
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(blob_name)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        download_stream = blob_client.download_blob()
        temp_file.write(download_stream.readall())
        temp_path = temp_file.name

    return temp_path


# Example usage
if __name__ == "__main__":
    # Upload example
    sample_text = "This is a sample document text to be stored in Azure Blob Storage."
    file_name = "sample_doc.txt"

    status = upload_blob(sample_text.encode('utf-8'), file_name)
    if status:
        print(f"File '{file_name}' uploaded successfully to Azure Blob Storage.")
    else:
        print("Failed to upload file.")

    # Download example
    test_blob_url = f"{AZURE_BLOB_ACCOUNT_URL}/{CONTAINER_NAME}/sample_doc.txt"
    local_file_path = download_blob_to_local(test_blob_url)
    print(f"Blob downloaded to local path: {local_file_path}")
