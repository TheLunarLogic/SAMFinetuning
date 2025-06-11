import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from Commons.FileService import cdn
from google.cloud import firestore

class Labellerr():
    def __init__(self):
        self.__db = firestore.Client()

    def download_files_by_statuses(self, project_id: str, statuses: list[str] = []):
        ids = []
        #logger.info("Getting IDs for audios in progress...")
        try:
            docs = self.__db.collection('projects').document(project_id).collection("images").where("status", 'in', statuses)
            snapshot = docs.stream()
            for doc in snapshot:
                doc_data = doc.to_dict()
                ids.append(doc_data["file_id"])
            return ids
        except  Exception as err:
            raise err
    
    def download_files_by_id(self, project_id: str, file_ids: list[str] = [], output_dir: str = ""):
        os.makedirs(output_dir, exist_ok=True)
        def download_file(file_id):
            if os.path.exists(f"{output_dir}/{file_id}.png"):
                return
            for attempt in range(3):
                try:
                    url = cdn.get_file_link(project_id, file_id)
                    file_path = f"{output_dir}/{file_id}.png"
                    urllib.request.urlretrieve(url, file_path)
                    break
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise e
                    continue
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(download_file, file_ids), total=len(file_ids), desc="Downloading files"))\
        
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory of the script directory
    parent_dir = os.path.dirname(script_dir)
    # Create output directory in the parent directory
    output_dir = os.path.join(parent_dir, "Images2")
    
    labellerr = Labellerr()
    file_ids = labellerr.download_files_by_statuses("marlee_firm_boa_46171", ["client_review"])
    labellerr.download_files_by_id("marlee_firm_boa_461711", file_ids, output_dir)

