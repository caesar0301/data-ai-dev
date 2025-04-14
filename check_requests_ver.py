import ray
import requests

@ray.remote
def get_requests_version():
    return requests.__version__

# Note: No need to specify the runtime_env in ray.init() in the driver script.
ray.init()
print("requests version:", ray.get(get_requests_version.remote()))
