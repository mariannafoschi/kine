=============
API Reference
=============

.. currentmodule:: kine

This page provides a complete reference for the public ``kine`` API. The
package is organized into five modules, each documented in its own section
below:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Description
   * - :ref:`kine.model`
     - Neural fields and learnable parameter modules.
   * - :ref:`kine.obsdata`
     - Data handling, extending ``ehtim``'s ``Obsdata`` class.
   * - :ref:`kine.trainer`
     - Training state and training-step driver.
   * - :ref:`kine.video`
     - Video object, plotting, and export utilities.
   * - :ref:`kine.utils`
     - Helpers for grids, batching, schedules, and I/O.

.. toctree::
   :maxdepth: 2

   model
   obsdata
   trainer
   video
   utils