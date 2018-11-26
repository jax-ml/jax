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
    sha256 = "599e9aad221a27882fce98ff472372030a2ebfe63cfc643d3470691f34bb68d6",
    strip_prefix="tensorflow-64e084b8cb27e8c53b15468c21f1b3471b4b9659",
    urls = [
      "https://github.com/tensorflow/tensorflow/archive/64e084b8cb27e8c53b15468c21f1b3471b4b9659.tar.gz",
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
