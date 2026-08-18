===========
Quick Start
===========

Minimal examples
----------------

The fastest way to try out ``kine`` is  to upload one of example notebooks from  
``notebooks/`` into `Google Colab <https://colab.research.google.com>`_ and 
follow the notebook's instructions for imaging the example datasets provided in 
``data/``.
Running the notebooks on Colab is a good way to get acquainted with ``kine`` 
without the need for installation and can be used for short test runs on small 
datasets. For larger datasets and longer runs we recommend installation on a GPU 
equipped local machine.

Running the code
----------------

Once ``kine`` is installed, it is possible to run one of the examples codes in 
``scripts/``. We provide examples for static, dynamic, multiepoch, polarimetric, 
and spectral imaging.

The code can be run on a single observation file with:

.. code-block:: bash

   python example_script.py -obs path/to/observations.uvfits -yml path/to/parameters.yml

and on multiple observation files, with:

.. code-block:: bash

   python example_script.py -obs path/to/obs/folder/ -yml path/to/parameters.yml

Alternatively ``kine`` can be run through the provided bash wrapper

.. code-block:: bash

   bash run_kine.sh

which assumes the same folder structure as the repository.

**Input requirements:**

- A UV-FITS file containing the interferometric observation.
- A YAML parameter file specifying imaging settings (see :doc:`parameters`).

**Output files:**

- Diagnostic PNG plots updating every few iterations, showing reconstructed frames and loss curves.
- Animated GIF of the reconstructed video.
- FITS or HDF5 file containing the final image, video, or spectral cube.
- Text file with fitted telescope gains.

