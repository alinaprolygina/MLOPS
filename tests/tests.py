import pytest
import numpy as np
import grpc
import mops.backend.apps.grpc.server_pb2 as server_proto
from mops.backend.apps.grpc.server_pb2_grpc import MopsAppStub
from mops.backend.managers import DatasetManager
from mops.frontend.managers.client import Client 

@pytest.mark.parametrize("name,X,y", [("test_dataset", [[1, 1], [0, 0], [-1, -1]], [1, 0, 1])])
def test_create_dataset(name, X, y, get_managers):
    DatasetManager.create_dataset(name, X, y)
    assert name in DatasetManager.list_available()

    response = DatasetManager.s3_client.list_objects_v2(Bucket=DatasetManager.bucket_name)
    all_names = [obj["Key"].split("/")[0] for obj in response["Contents"]]
    assert name in all_names

@pytest.mark.parametrize("name,X,y", [("test_dataset", [[1, 1], [0, 0], [-1, -1]], [1, 0, 1])])
def test_data_types(name, X, y, get_managers):
    X_get, y_get = DatasetManager.load_dataset(name)

    assert isinstance(X_get, np.ndarray)
    assert isinstance(y_get, np.ndarray)

    assert np.all(np.array(X) == X_get)
    assert np.all(np.array(y) == y_get)

@pytest.mark.parametrize("name", [("test_dataset")])
def test_delete(name, get_managers):
    DatasetManager.delete_dataset(name)
    assert name not in DatasetManager.list_available()
    
    response = DatasetManager.s3_client.list_objects_v2(Bucket=DatasetManager.bucket_name)
    all_names = [obj["Key"].split("/")[0] for obj in response["Contents"]]
    assert name not in all_names
