from google.cloud import storage
from google.cloud import storage

# Explicitly use service account credentials by specifying the key file path
storage_client = storage.Client.from_service_account_json("C:/Users/SAHAYOGA/Downloads/heartdisease-400219-ac537540c1e0.json")
)

# The rest of your code...

# Initialize a client
storage_client = storage.Client()


# Specify the bucket and file name
bucket_name = 'dataset0111'
blob_name = '/data/files/md5/f8/376ac9db4d25345aead44787474f27'

 

# Retrieve the blob (file) from GCS
bucket = storage_client.get_bucket(bucket_name)
blob = bucket.blob(blob_name)

 

# Download the blob to a local file
blob.download_to_filename('local-file.txt')