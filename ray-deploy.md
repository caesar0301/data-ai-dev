# Start kuberay operator

colima start --edit
colima stop

helm repo add ray https://ray-project.github.io/kuberay-helm/
helm repo update

helm search repo ray/kuberay-operator --versions
helm install ray-operator ray/kuberay-operator

# Latest ray version 2.44.1 (2025.04)

helm search repo ray/ray-cluster --versions

// chat values: https://github.com/ray-project/kuberay-helm/blob/main/helm-chart/ray-cluster/values.yaml

helm install myray ray/ray-cluster --version 1.3.2 \
  --set image.repository=registry.cn-hangzhou.aliyuncs.com/lacogito/ray \
  --set image.tag=2.44.1-py311-numpy2x \
  --set image.pullPolicy=Always

helm install myray ray/ray-cluster --version 1.3.2 \
  --set image.repository=rayproject/ray \
  --set image.tag=2.44.1-py311

kubectl port-forward service/myray-kuberay-head-svc 8265:8265 10001:10001

helm uninstall myray
helm uninstall kuberay-operator

# Check ray-python version compatibility

kubectl exec -it  <head-pod> -- python -c "import ray; print(ray.version); import sys; print(sys.version)"

pip install "ray[default,data]==2.44.1" "numpy<2.0"