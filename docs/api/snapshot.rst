Snapshots
=========

A snapshot is a minimal-memory representations of :cpp:concept:`bvh::ContactEntity`. Some :cpp:concept:`ContactEntities <bvh::ContactEntity>`
may have other data defined apart from that used by the BVH collision algorithms. In those cases, it is useful to define
snapshots of those entities in order to reduce the cost of copying, transferring, or migrating the data that is not
necessary for contact.

An additional advantage to snapshots is that they are serializable (in fact they are byte serializable), so when using
snapshots you do not have to specify a DARMA/vt serialization function for your entities.

Class ``entity_snapshot``
-------------------------

``#include <bvh/snapshot.hpp>``

.. doxygenstruct:: bvh::entity_snapshot
    :members:

Free Functions
^^^^^^^^^^^^^^

.. doxygenfunction:: bvh::make_snapshot