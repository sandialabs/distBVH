Bounding Volume Hierarchies
===========================

.. contents::

Entities
--------

The :cpp:concept:`ContactEntity` concept refers to data that can be stored in the bounding volume hierarchy. Since
concepts only exist in C++20, this can be viewed just as a set of requirements on entity types (such as elements or nodes)
that can be collided using BVH.

.. cpp:concept:: template< typename Entity > bvh::ContactEntity

    Any type of entity that can be collided using bounding volume hierarchies.

    The concept ``ContactEntity`` is satisfied if Entity defines the following member functions:

    - ``kdop()`` to return the :math:`k`-DOP bounds of the entity
    - ``centroid()`` to return the centroid of the entity
    - ``global_id()`` to return the unique index of the entity over all ranks

    Alternatively, the concept is satisfied if the following free functions can be applied to Entity via ADL:

    - ``get_entity_kdop( Entity )``
    - ``get_entity_centroid( Entity )``
    - ``get_entity_global_id( Entity )``

    That is, the following must be valid code:

    .. code-block::

        void test_func( const Entity &_e )
        {
            using namespace bvh;
            auto kdop = get_entity_kdop( _e );
            auto cent = get_entity_centroid( _e );
            auto id = get_global_id( _e );
        }


Class ``bvh_node``
------------------

``#include <bvh/node.hpp>``

.. doxygenclass:: bvh::bvh_node
    :members:

Free Functions
^^^^^^^^^^^^^^

.. doxygenfunction:: bvh::dump_node

Class ``bvh_tree``
------------------

``#include <bvh/tree.hpp>``

.. doxygenclass:: bvh::bvh_tree
    :members:

Type Aliases
^^^^^^^^^^^^

.. doxygentypedef:: bvh::bvh_tree_26d

.. doxygentypedef:: bvh::snapshot_tree_26d

Free Functions
^^^^^^^^^^^^^^

.. doxygenfunction:: bvh::dump_tree

Tree Construction
-----------------

``#include <bvh/tree_build.hpp>``

BVH provides several generic functions for constructing trees of various types. These have flexible tree-building
policies so new algorithms can be substituted in.

Shared-memory tree building
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Shared-memory tree building requires the entire tree to be stored on-node. There are two main algorithms BVH supports
for tree building on a shared-memory node: :cpp:class:`bottom-up <bvh::bottom_up_serial_builder>` and
:cpp:class:`top-down <bvh::top_down_builder>` building.

.. doxygenfunction:: bvh::rebuild_tree

.. doxygenfunction:: build_tree(span<const Element>)

.. doxygenfunction:: build_tree_top_down(span<const Element>)

.. doxygenfunction:: build_tree_top_down(const Container&)

.. doxygenfunction:: build_tree_bottom_up_serial(span<const Element>)

.. doxygenfunction:: build_tree_bottom_up_serial(const Container&)

These functions all provide techniques for constructing trees from containers of elements or spans of elements.
Note that the internal storage type of the tree is constructed from the given element (typically if the internal
storage type and the element type are the same this copies the element.

From an efficiency perspective it is more desirable to construct the tree from a snapshot (see :doc:`snapshot`).

.. doxygenfunction:: build_snapshot_tree_top_down(span<Element>)

.. doxygenfunction:: build_snapshot_tree_top_down(const Container&)

.. doxygenfunction:: build_snapshot_tree_bottom_up_serial(const Container&)

Usage Example
-------------

Typically tree building function and manipulation is carried out in shared-memory contexts (such as narrowphase kernels)
or when implementing a new distributed algorithm.

.. code-block::

    struct Element
    {
      bvh::dop26_d kdop();
      bvh::m::vec3d centroid();
      std::size_t global_id();
    };

    // ...

    std::vector< Element > elements = { /*...*/ };
    auto tree = build_snapshot_tree_top_down( elements );

    Entity test = { /*...*/ };

    collision_query_result< std::size_t > ret;
    detail::get_tree_overlapping_indices( test.kdop(), test.global_id(), tree,
                                          std::back_inserter( ret.pairs ) );






