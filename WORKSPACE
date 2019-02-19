load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
    ],
)

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
   name = "org_tensorflow",
   sha256 = "6c71ae56217b49c23ee1d76f6e8679d4e2ad050de325c9b3e1858dcce120a6e5",
   strip_prefix = "tensorflow-dafd60bab8ba61717ab4d431ffc6669f1e8c013e",
   urls = [
       "https://github.com/tensorflow/tensorflow/archive/dafd60bab8ba61717ab4d431ffc6669f1e8c013e.tar.gz",
   ],
)

# For development, one can use a local TF repository instead.
# local_repository(
#    name = "org_tensorflow",
#    path = "tensorflow",
# )

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)
