.. _type-promotion:

Type promotion semantics
========================

JAX's type promotion rules (i.e., the result of
:func:`jax.numpy.promote_types` for each pair of types) are given by the
following table, where, for example

* "b1" means :code:`np.bool_`,
* "s2" means :code:`np.int16`,
* "u4" means :code:`np.uint32`,
* "bf" means :code:`np.bfloat16`,
* "f2" means :code:`np.float16`, and
* "c8" means :code:`np.complex128`.

.. raw:: html

    <style>
        #types table {
          border: 2px solid #aaa;
        }

        #types td, #types th {
          border: 1px solid #ddd;
          padding: 3px;
        }
        #types th {
          border-bottom: 1px solid #aaa;
        }
        #types tr:nth-child(even){background-color: #f2f2f2;}
        #types .d {
          background-color: #ccf2cc;
        }
        #types td:first-child{
          background-color: #f2f2f2;
          border-right: 1px solid #aaa;
          font-weight: bold;
        }
        #types tr:first-child{background-color: #f2f2f2;}
    </style>

    <table id="types">
    <tr><th></th><th>b1</th><th>s1</th><th>s2</th><th>s4</th><th>s8</th><th>u1</th><th>u2</th><th>u4</th><th>u8</th><th>bf</th><th>f2</th><th>f4</th><th>f8</th><th>c4</th><th>c8</th></tr>
    <tr><td>b1</td><td>b1</td><td>s1</td><td>s2</td><td>s4</td><td>s8</td><td>u1</td><td>u2</td><td>u4</td><td>u8</td><td class="d">bf</td><td>f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>s1</td><td>s1</td]]><td>s1</td><td>s2</td><td>s4</td><td>s8</td><td>s2</td><td>s4</td><td>s8</td><td>f8</td><td class="d">bf</td><td>f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>s2</td><td>s2</td><td>s2</td><td>s2</td><td>s4</td><td>s8</td><td>s2</td><td>s4</td><td>s8</td><td>f8</td><td class="d">bf</td><td class="d">f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>s4</td><td>s4</td><td>s4</td><td>s4</td><td>s4</td><td>s8</td><td>s4</td><td>s4</td><td>s8</td><td>f8</td><td class="d">bf</td><td class="d">f2</td><td class="d">f4</td><td>f8</td><td class="d">c4</td><td>c8</td></tr>
    <tr><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>f8</td><td class="d">bf</td><td class="d">f2</td><td class="d">f4</td><td>f8</td><td class="d">c4</td><td>c8</td></tr>
    <tr><td>u1</td><td>u1</td><td>s2</td><td>s2</td><td>s4</td><td>s8</td><td>u1</td><td>u2</td><td>u4</td><td>u8</td><td class="d">bf</td><td>f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>u2</td><td>u2</td><td>s4</td><td>s4</td><td>s4</td><td>s8</td><td>u2</td><td>u2</td><td>u4</td><td>u8</td><td class="d">bf</td><td class="d">f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>u4</td><td>u4</td><td>s8</td><td>s8</td><td>s8</td><td>s8</td><td>u4</td><td>u4</td><td>u4</td><td>u8</td><td class="d">bf</td><td class="d">f2</td><td class="d">f4</td><td>f8</td><td class="d">c4</td><td>c8</td></tr>
    <tr><td>u8</td><td>u8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>u8</td><td>u8</td><td>u8</td><td>u8</td><td class="d">bf</td><td class="d">f2</td><td class="d">f4</td><td>f8</td><td class="d">c4</td><td>c8</td></tr>
    <tr class="d"><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td>bf</td><td class="d">bf</td><td>bf</td><td>f4</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>f2</td><td>f2</td><td>f2</td><td class="d">f2</td><td class="d">f2</td><td class="d">f2</td><td>f2</td><td class="d">f2</td><td class="d">f2</td><td class="d">f2</td><td class="d">f4</td><td>f2</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>f4</td><td>f4</td><td>f4</td><td>f4</td><td class="d">f4</td><td class="d">f4</td><td>f4</td><td>f4</td><td class="d">f4</td><td class="d">f4</td><td class="d">f4</td><td>f4</td><td>f4</td><td>f8</td><td>c4</td><td>c8</td></tr>
    <tr><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td>f8</td><td class="d">f8</td><td>f8</td><td>f8</td><td>f8</td><td>c8</td><td>c8</td></tr>
    <tr><td>c4</td><td>c4</td><td>c4</td><td>c4</td><td class="d">c4</td><td class="d">c4</td><td>c4</td><td>c4</td><td class="d">c4</td><td class="d">c4</td><td class="d">c4</td><td>c4</td><td>c4</td><td>c8</td><td>c4</td><td>c8</td></tr>
    <tr><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td class="d">c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td><td>c8</td></tr>
    </table><p>

Jax's type promotion rules differ from those of NumPy, as given by
:func:`numpy.promote_types`, in those cells highlighted with a green background
in the table above. There are two key differences:

* when promoting an integer or boolean type against a floating-point or complex
  type, JAX always prefers the type of the floating-point or complex type.

  Accelerator devices, such as GPUs and TPUs, either pay a significant
  performance penalty to use 64-bit floating point types (GPUs) or do not
  support 64-bit floating point types at all (TPUs). Classic NumPy's promotion
  rules are too willing to overpromote to 64-bit types, which is problematic for
  a system designed to run on accelerators.

  JAX uses floating point promotion rules that are more suited to modern
  accelerator devices and are less aggressive about promoting floating point
  types. The promotion rules used by JAX for floating-point types are similar to
  those used by PyTorch.

* JAX supports the
  `bfloat16 <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_
  non-standard 16-bit floating point type
  (:code:`jax.numpy.bfloat16`), which is useful for neural network training.
  For a description of bfloat16, see details
  The only notable promotion behavior is with respect to IEEE-754
  :code:`float16`, which which :code:`bfloat16` promotes to a :code:`float32`.