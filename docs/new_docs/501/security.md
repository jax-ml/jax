---
nosearch: true
---

(jax-501-security)=
# Security considerations

<!--* freshness: { reviewed: '2026-08-17' } *-->

This page describes features for securing your JAX deployment.

(jax-501-coordination-service-mtls)=
## Securing the JAX coordination service

### Basic setup

{func}`jax.distributed.initialize` starts a gRPC coordination service on process 0, and
every process connects to it as a client. By default, these connections are neither
encrypted nor authenticated. If insecure connections are used and an attacker is able to
reach the `coordinator_address`, pose as the coordinator, or intercept the network
traffic, they could observe or modify cluster bringup, JAX's key/value store, and other
service functionality. Protections outside of JAX include firewall rules, Kubernetes
network policies, and GCP's [automatic encryption in
transit](https://docs.cloud.google.com/docs/security/encryption-in-transit#encryption-transit-with-networks).

If you would like to directly secure the coordination service connections,
`jax.distributed.initialize` has optional arguments and corresponding configs for
enabling mutual TLS (mTLS). mTLS means that the client and service are both
authenticated to each other by each presenting a certificate signed by a certificate
authority (CA) verifying their identity, and encrypt subsequent traffic. (TLS is the
same except only the server is authenticated; any client can connect.) In other words,
for each connection to the service, both the client and service authenticate their
corresponding "peer" (the service or client, respectively).

To enable mTLS, all of the following arguments must be provided:
* `mtls_cert_file` (or env var `JAX_MTLS_CERT_FILE`): the process' signed certificate it
  got from the CA that it'll send to its peer process
* `mtls_key_file` (or env var `JAX_MTLS_KEY_FILE`): the process' private key, used to
  prove to its peer that it owns `mtls_cert_file`
* `mtls_ca_file` (or env var `JAX_MTLS_CA_FILE`): the CA's certificate, used to verify the
  `mtls_cert_file` from its peer process

Setting up these files is beyond the scope of this document.

`verify_secure_credentials` (or env var
`JAX_DISTRIBUTED_VERIFY_SECURE_CREDENTIALS`) can optionally be set to `True` to
crash if none of the above mTLS arguments are provided, meaning insecure connections
would be used. Setting any of the mTLS arguments causes JAX to raise an exception if
mTLS isn't properly configured, regardless of `verify_secure_credentials`.

### Peer identity verification

You may optionally provide `mtls_peer_uri_prefix` (or env var
`JAX_MTLS_PEER_URI_PREFIX`). This changes how each process decides if its peer's signed
identity is valid.

Without the prefix (default behavior), clients verify that the service both presents a
valid certificate signed by the CA, and that an identity in the certificate matches the
service's hostname (i.e. `coordinator_address`). The service just verifies that clients
have a valid certificate signed by the CA, since the service doesn't have a list of
client addresses to expect. This is sufficient for many deployments with dedicated CAs,
where just obtaining a signed certificate from the CA means the client can be trusted.

With the prefix, the client and service both verify that their peer presents a valid
certificate, and that the identity (specifically any [URI
SAN](https://en.wikipedia.org/wiki/Public_key_certificate#Subject_Alternative_Name_certificate))
in the certificate starts with the prefix. The prefix check replaces the service
hostname check by the client. Specifying a prefix can be useful when the
`coordinator_address` doesn't match any identities in the certificate, or to provide
additional client verification by the service. For example, [SPIFFE](https://spiffe.io/)
deployments use a single CA across different workloads, in which case the prefix can be
used to make sure all connections are coming from the same workload.

The prefix is required to end with `/`. Otherwise a prefix like `spiffe://example.org`
would match intended identities like `spiffe://example.org/my-job/`, but also unintended
ones like `spiffe://example.org.evil/...`.

## Securing TPU services

The TPU runtime starts its own internal services, e.g. for bootstrapping multi-host TPU
topologies and handling multi-slice network collectives. As of libtpu 0.0.46, you can
enable mTLS for these services using the following env vars:

* `TLS_CERT_FILE`: the process' signed certificate
* `TLS_KEY_FILE`: the process' private key
* `TLS_CA_FILE`: the CA's certificate
* `TLS_VERIFIER`: `noop`, `subject_prefix`, or `cel` (default `noop`). `noop` means both
  sides check for a valid certificate but don't do any identity verification.
* `TLS_VERIFIER_PREFIX`: the peer SAN prefix to require if
  `TLS_VERIFIER=subject_prefix`. Unlike `JAX_MTLS_PEER_URI_PREFIX`, this covers the
  common name and all SAN types, not just URI SANs.
* `TLS_VERIFIER_CEL_EXPRESSION`: a [Common Expression Language](https://cel.dev/)
  expression over variables `common_name`, `dns_names`, `uri_names`, `ip_names`, and
  `email_names` that returns true if the peer's identity should be accepted

See the above section for more information on mTLS and how prefix verification works.

## Other network connections

The following network connections are plaintext and unauthenticated. JAX doesn't have
security features for these, so consider external options such as firewall rules,
Kubernetes network policies, and GCP's [automatic encryption in
transit](https://docs.cloud.google.com/docs/security/encryption-in-transit#encryption-transit-with-networks).

* CPU and non-NVLink GPU collectives
* Connections to the server started by {func}`jax.profiler.start_server`, which can be
  used to capture profiles of the running workload
