"""
This script uploads a machine learning model file to Azure Blob Storage, creating a new version
of the model if previous versions exist.

Modules:
    - os: Provides functions to interact with the operating system.
    - azure.storage.blob: Contains classes for interacting with Azure Blob Storage.

Environment Variables:
    - AZURE_STORAGE_CONNECTION_STRING: The connection string for the Azure Storage account.

Constants:
    - model_file: The relative path to the model file to be uploaded.
    - container_name: The name of the Azure Blob Storage container.

Workflow:
    1. Checks if the model file exists locally. If not, raises a FileNotFoundError.
    2. Attempts to create the specified Azure Blob Storage container. If it already exists, the exception is ignored.
    3. Lists existing blobs in the container to determine the latest version of the model.
    4. Calculates the new version number for the model.
    5. Uploads the model file to Azure Blob Storage with the new version number in the filename.
    6. Prints a success message indicating the version number of the uploaded model.

Exceptions:
    - FileNotFoundError: Raised if the specified model file does not exist.
"""
import os
from azure.storage.blob import BlobServiceClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

model_file = "model/immoscout_model.pkl"
container_name = "immoscout-models"

logger.info(f"🔍 Checking if the file '{model_file}' exists...")
if not os.path.exists(model_file):
    logger.error(f"❌ The file '{model_file}' was not found. Ensure the model is saved.")
    raise FileNotFoundError(f"The file '{model_file}' was not found.")

try:
    blob_service_client.create_container(container_name)
    logger.info(f"✅ Container '{container_name}' created.")
except Exception:
    logger.info(f"ℹ️ Container '{container_name}' already exists.")

blobs = blob_service_client.get_container_client(container_name).list_blobs()
existing_versions = [
    int(blob.name.split("-")[-1].replace(".pkl", ""))
    for blob in blobs
    if blob.name.startswith("immoscout-model") and blob.name.endswith(".pkl")
]
new_version = max(existing_versions, default=0) + 1
logger.info(f"✅ New model version will be: {new_version}")

blob_client = blob_service_client.get_blob_client(
    container=container_name, blob=f"immoscout-model-{new_version}.pkl"
)
with open(model_file, "rb") as data:
    blob_client.upload_blob(data)

logger.info(f"✅ Model uploaded as version {new_version} to Azure Blob Storage!")