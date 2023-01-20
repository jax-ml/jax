load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "org_tensorflow",
    sha256 = "dc7063605dc5281b6b99b1b0b2de250f55bd7fb959b27a86d2f91cae0528fd5d",
    strip_prefix = "tensorflow-48f9d3dfcf148e1c2dfcf79b6334c2e2a2783093",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/48f9d3dfcf148e1c2dfcf79b6334c2e2a2783093.tar.gz",
    ],
)

# For development, one often wants to make changes to the TF repository as well
# as the JAX repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the TF repository on the build.py command line by passing a flag
#    like:
#    python build/build.py --bazel_options=--override_repository=org_tensorflow=/path/to/tensorflow
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "org_tensorflow",
#    path = "/path/to/tensorflow",
# )

load("//third_party/ducc:workspace.bzl", ducc = "repo")
ducc()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()
