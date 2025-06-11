import grpc

# import the generated classes
import Commons.FileService.cdn_pb2 as cdn_pb2
import Commons.FileService.cdn_pb2_grpc as cdn_pb2_grpc
from Commons.FileService import AuthToken
# open a gRPC channel
import os
import config
channel = grpc.secure_channel(config.cdn_server_address, grpc.ssl_channel_credentials())

# create a stub (client)
stub = cdn_pb2_grpc.CDNStub(channel)

def get_file_link(project_id, file_id):
    """ fetches file link from cdn """

    metadata = []
    signed_jwt = AuthToken.generateIdToken()
    metadata.append(("authorization", "Bearer " + signed_jwt))
    # create a valid request message
    request = cdn_pb2.FileLink(project_id=project_id,
        file_id=file_id)

    # make the call
    response = stub.GetFileLink(request, metadata=metadata)

    # et voilà
    return response.url

if __name__ == "__main__":
    print(get_file_link("ann_adequate_swordtail_52996", "05bdfa93-e213-489d-b10f-82a0db1a969b"))