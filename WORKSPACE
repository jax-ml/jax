load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update XLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "xla",
    sha256 = "60558f5c2a2ff46f46194febdc808180b6536a9d42b2835cd738afff462a496b",
    strip_prefix = "xla-81b357b48e9b1c9dc3381a6ab4a2a64b06c34fc7",
    urls = [
        "https://github.com/openxla/xla/archive/81b357b48e9b1c9dc3381a6ab4a2a64b06c34fc7.tar.gz",
    ],
)

# For development, one often wants to make changes to the TF repository as well
# as the JAX repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the TF repository on the build.py command line by passing a flag
#    like:
#    python build/build.py --bazel_options=--override_repository=xla=/path/to/xla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "xla",
#    path = "/path/to/xla",
# )

load("//third_party/ducc:workspace.bzl", ducc = "repo")
ducc()

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()


load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()
