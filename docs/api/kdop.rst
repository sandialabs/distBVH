*k*-DOPS
========

.. contents::

The bvh::kdop_base class represents a basic :math:`k`-DOP, or Discrete Oriented Polytope. This represents a volume of space
bounded by :math:`k` discretely oriented bounding slabs. A common example of this is the axis-aligned bounding box
(AABB), but this can be generalized to :math:`k` sided polytope.

The bounding slabs of a :math:`k`-DOP are a fixed orientation. In two dimensions, a :math:`4`-DOP is a bounding box with
the 4 axes being the cardinal axes and their negative directions. An :math:`8`-DOP is bounded by the 4 primary axes in
addition to 4 additional axes representing the diagonals. This generates an octagon with fixed angles between edges.

Generalizing to 3 dimensions, a :math:`6`-DOP is an AABB. BVH provides an implementation of a :math:`26`-DOP
(in the class bvh::dop_26). This can be imagined as a cube (6 axes) with the corners sliced off (8 additional axes) and
the edges sliced off (12 additional axes).

Class ``extent``
----------------

``#include <bvh/kdop.hpp>``

.. doxygenstruct:: bvh::extent
    :members:

Class ``kdop_base``
-------------------

``#include <bvh/kdop.hpp>``

.. doxygenstruct:: bvh::kdop_base
    :members:

Free Functions
^^^^^^^^^^^^^^

.. doxygenfunction:: bvh::overlap(const kdop_base<T, K, Derived>&, const kdop_base<T, K, Derived>&)

.. doxygenfunction:: bvh::merge(const kdop_base<T, K, Derived>&, const kdop_base<T, K, Derived>&)

:math:`k`-DOP Presets
---------------------

BVH provides various useful presets for :math:`k`-DOPs:

.. doxygenstruct:: bvh::dop_6

.. doxygenstruct:: bvh::dop_18

.. doxygenstruct:: bvh::dop_26

Type Aliases
------------

.. doxygentypedef:: dop_6d

.. doxygentypedef:: dop_18d

.. doxygentypedef:: dop_26d

Using the :math:`k`-DOP class
-----------------------------

:cpp:class:`~template\<typename T, int K, typename Derived> bvh::kdop_base` is a versatile class that provides basic
functionality for all :math:`k`-DOP types, including both preset or user-defined.

Constructing :math:`k`-DOPs
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default constructor of :cpp:class:`~template\<typename T, int K, typename Derived> bvh::kdop_base` initializes each dimension
of the :math:`k`-DOP to a zero-length extent.

To construct a :cpp:class:`~template\<typename T, int K, typename Derived> bvh::kdop_base` from vertices, a vertex
class must overload ``operator[]``:

.. code-block::

    using Vertex = std::array< double, 3 >;

Then a range of iterators can just be passed in (see :cpp:func:`~bvh::kdop_base::from_vertices()`):

.. code-block::

    std::vector< Vertex > vertices = ...
    auto kdop = bvh::dop_26::from_vertices( vertices.begin(), vertices.end() );

Additionally, the bounding volume can be expanded:

.. code-block::

    auto kdop_expanded = bvh::dop_26::from_vertices( vertices.begin(), vertices.end(), 0.01 );

Where the last parameter is the amount of expansion (defaults to ``0.0``).


The :math:`k`-DOP can also be constructed by a sphere (see :cpp:func:`~bvh::kdop_base::from_sphere()`):

.. code-block::

    auto kdop = bvh::dop_26::from_sphere( 0.0, 0.0, 0.0, 5.0 );


:cpp:func:`~bvh::kdop_base::from_sphere()` is identical to calling :cpp:func:`~bvh::kdop_base::from_vertices()` with a single
vertex and an epsilon.

User-defined :math:`k`-DOPs
---------------------------

In some cases, a user may want to define their own :math:`k`-DOP classes. This may be useful in situations where higher
or lower values for :math:`k` are required, or where the axes are non-standard.

In this case, inherit from :cpp:class:`~template\<typename T, int K, typename Derived> bvh::kdop_base` using the
Curiously-Recurring Template Pattern (CTRP), providing the desired number of axes and the extent arithmetic type:

.. code-block::

    // Create a dop with 17 axes
    template< typename T >
    struct my_dop : public kdop_base< T, 17, my_dop >
    {
      using typename kdop_base< T, 17, my_dop >::arithmetic_type;
      using kdop_base< T, 17, my_dop >::kdop_base;
      // ...
    };

The normals of the :math:`k`-DOP need to be defined in a static constexpr ``BVH_INLINE`` (if intended to be used in
CUDA/Kokkos kernels) member function called ``normals()``:

.. code-block::

    static constexpr BVH_INLINE array< m::constant_vec3< T >, 17 > normals()
    {
      return array< m::constant_vec3< T >, 17 >{{
        m::constant_vec3< T >( 1., 0., 0. ),
        m::constant_vec3< T >( 0., 1., 0. ),
        m::constant_vec3< T >( 0., 0., 1. ),
        // ... and so on
  }};

Note that the number of elements in this array *must* be equal to the number specified for :math:`k`.