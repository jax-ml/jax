:orphan:

.. _beginner-guide:


Beginner Guide
==============

Welcome to the beginners guide for JAX. 
On this page we will introduce you to the key ideas of JAX,
show you how to get JAX running
and provide you some tutorials to get started.

If looking to jump straight in take a look at the JAX quickstart.

.. toctree::
   :maxdepth: 1

   notebooks/quickstart

For most users starting out the key functionalities of JAX to become familiar with are :code:`jax.jit`,
:code:`jax.grad`, and :code:`jax.vmap`. 
A good way to get familiar with this is with the Jax-101 tutorials.

.. toctree::
   :maxdepth: 2

   jax-101/index

If you're familiar with doing array-oriented computing with NumPy, you may find the following
resources useful:

.. toctree::
   :maxdepth: 1

   notebooks/thinking_in_jax
   notebooks/Common_Gotchas_in_JAX
   faq

If you prefer a video introduction here is one from JAX contributor Jake Vanderplas:

.. raw:: html

	<iframe width="640" height="360" src="https://www.youtube.com/embed/WdTeDXsOSj4"
	 title="Intro to JAX: Accelerating Machine Learning research"
	frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
	allowfullscreen></iframe>
