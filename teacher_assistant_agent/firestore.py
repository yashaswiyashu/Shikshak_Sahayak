import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import os
# Use a service account.
current_dir = os.path.dirname(os.path.abspath(__file__))
service_account_path = os.path.join(current_dir, 'firebaseServiceAccount.json')
cred = credentials.Certificate(service_account_path)

app = firebase_admin.initialize_app(cred)

db = firestore.client()