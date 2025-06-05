from pydantic import BaseModel
from datetime import datetime

class Metadata(BaseModel):
    experiment_id: str
    run_id: str
class Metrics(BaseModel):
    mAP50_95: float
class AutoLabelModel(BaseModel):
    access: str = 'public'
    algorithm_id: str
    project_id: str
    client_id: str
    metadata: Metadata
    metrics: Metrics
    model_name: str
    model_id: str
    job_id: str
    email_id: str
    data_type: str
    status: str = 'deployed'
    created_at: int = int(datetime.now().timestamp())
    created_by: str

class AutoLabelJob(BaseModel):
    job_id: str
    client_id: str
    project_id: str
    algorithm_id: str
    email_id: str
    data_type: str
    job_type: str
    job_name: str
    job_status: str = 'FINISHED'
    created_at: int = int(datetime.now().timestamp())
    created_by: str
    metadata: Metadata
    metrics: Metrics