

# Polymorphic python scalars

Like jax0 we 're doing a bit of overloading so you can write `2 * x` where `x`
can be a f32 or i64 or whatever. But this time we're not building it into the
type system ("weak types"). It's just done one-layer deep at the level of
individual ops.


# Primitive representations

In jax0 a primitive is just a name. Handlers for each interpreter get added one
by one to rule tables. The benefit of this style is that it gives you some
freedom in how to organize the rules: you can group them by primitive or by
interpreter depending on your preference. Here instead we use the conventional
Python way to represent "a bunch of functions that have to be implemented when
you add a new thing": classes. Each primitive op is a subclass of `Op` and the
standard interpretation rules are methods of that class. I think it's clearer,
more accessible, and it works better with existing tooling. And if you want to
organize rules by interpreter instead of by primitive, perhaps you're
implementing a new experimental interpreter for example, then you can do that
with monkey-patching.

