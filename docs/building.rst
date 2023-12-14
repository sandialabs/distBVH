Building
========

.. role:: shell(code)
    :language: sh

BVH requires at least `CMake`_ version 3.15 for building. Typically this is
invoked by creating a separate out-of-source build directory:

.. code-block:: sh

    cd bvh
    mkdir build && cd build
    cmake <cmake options> ..

The options passed to cmake depend on your desired build configuration. This manual will document the various options
that are summarized here:

- :ref:`vt_DIR <cmake-vt-dir>`
- :ref:`vt_ROOT <cmake-vt-root>`
- :ref:`Kokkos_DIR <cmake-kokkos-dir>`
- :ref:`Kokkos_ROOT <cmake-kokkos-root>`
- :ref:`BVH_VT_INSERTABLE_COLLECTIONS <cmake-bvh-vt-insertable-collections>`
- :ref:`VTK_ROOT <cmake-vtk-root>`
- :ref:`BVH_DEBUG_LEVEL <cmake-bvh-debug-level>`

After the cmake configure step has completed, build using the desired number of parallel processors:

.. code-block:: sh

    make -j <number of cores>


.. _CMake: https://cmake.org/

Locating DARMA/vt
-----------------

BVH is dependent on `DARMA/vt`_ for parallel asynchronous distributed tasking. Follow the instructions on the `DARMA/vt README`_ for
building VT and install it to a desired directory.

.. _cmake-vt-dir:

.. code-block:: sh

    -Dvt_DIR=path/to/vt/install/cmake

In more recent versions of cmake, the ``vt_ROOT`` variable can be used instead of ``vt_DIR``:

.. _cmake-vt-root:

.. code-block:: sh

    -Dvt_ROOT=path/to/vt/install

Other variables, like ``CMAKE_PREFIX_PATH`` work as expected.

.. _DARMA/VT: https://github.com/DARMA-tasking/vt
.. _DARMA/vt README: https://github.com/DARMA-tasking/vt/blob/develop/README.md

VT Insertable Collections
^^^^^^^^^^^^^^^^^^^^^^^^^

BVH can optionally use insertable collections. Use the following to turn off insertable collections (defaults to on):

.. _cmake-bvh-vt-insertable-collections:

.. code-block:: sh

    -DBVH_VT_INSERTABLE_COLLECTIONS=OFF

Locating Kokkos
-----------------

BVH is dependent on `Kokkos`_ for performance portability.
Follow the instructions on `Kokkos wiki`_ for Kokkos build and installation.

Use ``Kokkos_DIR`` or ``Kokkos_ROOT`` to point to the installation directory:

.. _cmake-kokkos-dir:

.. code-block:: sh

    -DKokkos_DIR=path/to/kokkos/install/cmake

.. _cmake-kokkos-root:

.. code-block:: sh

    -DKokkos_ROOT=path/to/kokkos/install

.. _Kokkos: https://github.com/kokkos/kokkos
.. _Kokkos wiki: https://kokkos.github.io/kokkos-core-wiki/building.html

Building with VTK
-----------------

BVH supports using `VTK`_ for visualizing bounding volume hierarchies. If using VTK, add the following option to the cmake
invocation:

.. _cmake-vtk-root:

.. code-block:: sh

    -DVTK_ROOT=path/to/vtk/install/

.. _VTK: https://vtk.org/

Debug output and traces
-----------------------

The debug output/tracing level of BVH can be configured at compile time. It is recommended to keep this low or at its
default (0) to avoid your stdout being flooded and reduced performance.

.. _cmake-bvh-debug-level:

.. code-block:: sh

    -DBVH_DEBUG_LEVEL=${DESIRED_DEBUG_LEVEL}

Building this documentation
---------------------------

This documentation uses a combination of `Sphinx`_ and `Breathe`_. These dependencies can be installed via
:shell:`pip install -r requirements.txt` on a relatively recent version of Python (e.g. Python 3.5), preferably in a
virtual environment.

Add the following to your cmake options:

.. code-block:: sh

    -DCMAKE_PROGRAM_PATH=/path/to/doxygen/bin
    -DSphinx_ROOT=/path/to/python/install/or/venv

If doxygen is installed in a standard location (e.g. /usr/local/bin) there is no need to specify ``CMAKE_PROGRAM_PATH``.

Then, build the *doc* target

.. code-block:: sh

    make doc

.. _Sphinx: https://www.sphinx-doc.org/
.. _Breathe: https://breathe.readthedocs.io/