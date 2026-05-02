=============
API Reference
=============

.. currentmodule:: kine

The ``kine`` package is organized into five modules:

- **obsdata** : the module is an expansion of ``eht-imaging``'s `Obsdata` class for manipulation of VLBI data. It inherits all attributes and methods from the original class, while adding new methods for data processing with ``kine``. An `obsdata` object contains data and metadata pertaining to a VLBI observation dataset.
- **model** : the module defines the neural networks that model the source and the learnable parameters that model complex gains.
- **video** : the module defines the 'video' class, which consists of a polarimetric video array and associated metadata. Method of the class include plotting and saving routines.
- **trainer** : the module defines the jax training state and the different loss function terms.
- **utils** : the module contains various helper functions, including functions related to optimization hyperparameters, domain coordinates, and data formatting.


.. list-table::
   :header-rows: 1
   :widths: 20 80
   :align: left

   * - Module
     - Description
   * - :doc:`kine.obsdata <obsdata>`
     - Data handling, extending ``ehtim``'s ``Obsdata`` class.
   * - :doc:`kine.model <model>`
     - Neural fields and learnable parameter modeling for source and gains.
   * - :doc:`kine.video <video>`
     - Video object, plotting and saving utilities.
   * - :doc:`kine.trainer <trainer>`
     - Training state and loss function terms.
   * - :doc:`kine.utils <utils>`
     - Various utilities, including grids, batching, schedules, data formatting.

.. toctree::
   :hidden:
   :maxdepth: 2

   obsdata
   model
   video
   trainer
   utils