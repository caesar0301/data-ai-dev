colima start --with-kubernetes --cpu 4 --memory 6 --disk 20

helm repo add ray https://ray-project.github.io/kuberay-helm/
helm repo update

helm search repo ray/kuberay-operator --versions
helm install ray-operator ray/kuberay-operator

# latest ray version 2.44.1 (2025.04)
helm search repo ray/ray-cluster --versions

helm install myray ray/ray-cluster --version 1.3.2 \
    --set image.repository=registry.cn-hangzhou.aliyuncs.com/lacogito/ray \
    --set image.tag=2.44.1-py311-numpy21

helm install myray ray/ray-cluster --version 1.3.2 \
    --set image.tag=2.44.1-py311

kubectl port-forward service/myray-kuberay-head-svc 8265:8265 10001:10001

helm uninstall myray
helm uninstall kuberay-operator

# check ray-python version compatibility

kubectl exec -it <ray-head-pod> -- python -c "import ray; print(ray.__version__); import sys; print(sys.version)"

# fix numpy versions

kubectl exec -it <ray-head-pod> -- python -c "import numpy; print(numpy.__version__); import numpy._core.numeric"
