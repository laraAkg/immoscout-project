import os
from azure.storage.blob import BlobServiceClient

# Verbindung zum Azure Blob Storage herstellen
connect_str = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

# Modell und Zielcontainer
model_file = "model/immoscout_model.pkl"
container_name = "immoscout-models"

# Überprüfen, ob die Datei existiert
if not os.path.exists(model_file):
    raise FileNotFoundError(f"❌ Die Datei '{model_file}' wurde nicht gefunden. Stelle sicher, dass das Modell gespeichert wurde.")

# Container erstellen, falls nicht vorhanden
try:
    blob_service_client.create_container(container_name)
except Exception:
    pass  # Container existiert evtl. schon

# Versionierung über aufsteigende Nummer
blobs = blob_service_client.get_container_client(container_name).list_blobs()
existing_versions = [
    int(blob.name.split('-')[-1].replace('.pkl', ''))
    for blob in blobs
    if blob.name.startswith("immoscout-model") and blob.name.endswith(".pkl")
]
new_version = max(existing_versions, default=0) + 1

# Upload des Modells
blob_client = blob_service_client.get_blob_client(container=container_name, blob=f"immoscout-model-{new_version}.pkl")
with open(model_file, "rb") as data:
    blob_client.upload_blob(data)

print(f"✅ Modell wurde als Version {new_version} nach Azure Blob Storage hochgeladen!")