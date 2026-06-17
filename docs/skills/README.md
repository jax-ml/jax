<!--*
freshness: {
  owner: "necula"
  reviewed: "2026-06-17"
}
*-->

To use these JAX skills, you have to configure your agent to find them.

E.g., for Google-internal usage, you could add
the following to `//depot/configs/users/<ldap>/_agents/skills.json`:
```
{
  "entries": [
    { "path": "<WORKSPACE_ROOT>/jax/docs/skills" }
  ]
}
```
