from jax._src import clusters

def test_cluster_gke_before_k8s():
    cluster_types = clusters.ClusterEnv._cluster_types
    print("ClusterEnv._cluster_types")
    print(*cluster_types, sep='\n')
    assert len(cluster_types) > 0

    # Ensure that GKE detection happens before K8s.
    class_names = [c.__name__ for c in cluster_types]
    assert "GkeTpuCluster" in class_names
    assert "K8sCluster" in class_names
    assert class_names.index("GkeTpuCluster") < class_names.index("K8sCluster")
