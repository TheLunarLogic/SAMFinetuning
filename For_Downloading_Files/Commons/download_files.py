from Base.Singleton import Singleton
from concurrent.futures import ThreadPoolExecutor
from Commons.FileService import cdn
import urllib.request
import os
from tqdm import tqdm
DEFAULTS = {
    'download_dir': 'downloads'
}
class Labellerr(Singleton):

    def download_files_by_statuses(self, project_id: str, statuses: list[str] = []):
        raise NotImplementedError("download_files_by_statuses is not implemented")

    def download_files_by_id(self, project_id: str, file_ids: list[str] = [], output_dir: str = ""):
        if output_dir == "":
            output_dir = DEFAULTS['download_dir']
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
            list(tqdm(executor.map(download_file, file_ids), total=len(file_ids), desc="Downloading files"))


if __name__ == "__main__":
    labellerr = Labellerr.get_instance()
    folder_path = "./kapsys-batch-2-export"
    file_ids = [f.split(".")[0] for f in os.listdir(folder_path)]

    labellerr.download_files_by_id("cahra_front_dog_45807", file_ids, output_dir="./kapsys-batch-2-files")
