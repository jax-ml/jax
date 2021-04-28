load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "org_tensorflow",
    sha256 = "4e07806f8786baa478d38e89974fbcf8bf9e41839f29b05f210fce02427b39d1",
    strip_prefix = "tensorflow-b0e85b5b3859d060a42364c79fe664b07299a0e9",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/b0e85b5b3859d060a42364c79fe664b07299a0e9.tar.gz",
    ],
)

# For development, one can use a local TF repository instead.
# local_repository(
#    name = "org_tensorflow",
#    path = "tensorflow",
# )

load("//third_party/pocketfft:workspace.bzl", pocketfft = "repo")
pocketfft()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")
workspace()
load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")
workspace()
