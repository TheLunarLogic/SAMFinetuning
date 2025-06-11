import subprocess
import datetime
import os
from google.cloud import firestore

def upload_model_and_update_firestore(client_id, source_file):
    """
    Uploads a model file to GCS and updates the Firestore document with the model path.
    
    Args:
        client_id: The client ID (workspace ID)
        source_file: Path to the model checkpoint file
    
    Returns:
        str: The GCS path where the model was uploaded
    """
    # Get current timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define paths
    bucket_name = "autolabel-models"
    gcs_path = f"workspaces/{client_id}/SAM"
    model_filename = f"{timestamp}.ckpt"
    full_gcs_path = f"gs://{bucket_name}/{gcs_path}/{model_filename}"

    # Upload the file to GCS
    print(f"Uploading model to {full_gcs_path}")
    try:
        subprocess.run(["gsutil", "cp", source_file, full_gcs_path], check=True)
        print("Upload complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error uploading file: {e}")
        return None
    
    # Format the path for Firestore (without gs:// prefix)
    firestore_path = f"{bucket_name}/{gcs_path}/{model_filename}"
    
    # Update Firestore
    try:
        # Initialize Firestore client
        db = firestore.Client()
        
        # Reference to the workspace document
        workspace_ref = db.collection('workspaces').document(client_id)
        
        # Update the SAM field
        workspace_ref.update({
                'SAM.checkpoint_path': firestore_path
            })
        
        print(f"Updated Firestore document for client {client_id} with SAM model path: {firestore_path}")
        return firestore_path
    
    except Exception as e:
        print(f"Error updating Firestore: {e}")
        return None

if __name__ == "__main__":
    # Replace with your actual client ID
    client_id = "79"
    
    # Path to the model checkpoint
    source_file = "sam_lora_checkpoint/sam-lora-epoch=29-val_iou=0.8472.ckpt"
    
    # Upload model and update Firestore
    result = upload_model_and_update_firestore(client_id, source_file)
    
    if result:
        print(f"Process completed successfully. Model path: {result}")
    else:
        print("Process failed. Check the error messages above.")