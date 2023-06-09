import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase/isboozed-firebase-adminsdk-bnfo5-e7d3c65002.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'isboozed.appspot.com'
})

# Create a storage client
bucket = storage.bucket()

# Specify the file you want to download
file_name = 'audio/recording-ce2bec63-31eb-4035-b914-40039e84ad8e.wav'

# Download the file
blob = bucket.blob(file_name)
blob.download_to_filename('downloads/recording-ce2bec63-31eb-4035-b914-40039e84ad8e.x-wav')

print('File downloaded successfully.')
