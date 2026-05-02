=============
API Reference
=============

.. currentmodule:: kine

The ``kine`` package is organized into five modules:

- **obsdata** : the class is an expansion of ``eht-imaging``'s Obsdata class for manipulation of VLBI data. It inherits all attributes and methods from ehtim's class, while adding new methods for data processing with ``kine``. An `obsdata` object contains 
- ``model`` : 
- ``video`` : 
- ``trainer`` : 
- ``utils`` : 


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