# Github Actions workflows

See the Github documentation for more information on Github Actions in general.

## Notes

* <https://opensource.google/documentation/reference/github/services#actions>
  mandates using a specific commit for non-Google actions. We use
  [Ratchet](https://github.com/sethvargo/ratchet) to pin specific versions.  If
  you'd like to update an action, you can write something like `uses:
  'actions/checkout@v4'`, and then run `./ratchet pin workflow.yml` to convert
  to a commit hash. See the Ratchet README for installation and more detailed
  instructions.
