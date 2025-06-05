
from google.cloud import firestore

client = firestore.Client()

def get_questions(project_id: str):
    questions = client.collection('projects').document(project_id).collection('questions').get()
    return [e.to_dict() for e in questions]