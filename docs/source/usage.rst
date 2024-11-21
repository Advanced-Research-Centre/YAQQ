Usage
=====

.. _installation:

Installation
------------

To use YAQQ, first install it using pip:

.. code-block:: console

   (.venv) $ pip install yaqq

Manual Mode
----------------

.. code-block:: console

   from yaqq import yaqq
   qq = yaqq()
   qq.yaqq_manual()

API Mode
----------------

.. code-block:: console

   from yaqq import yaqq
   qq = yaqq()
   qq.yaqq_cfg(CONFIG)