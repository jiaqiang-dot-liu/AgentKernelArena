.. meta::
   :description: AgentKernelArena is a standardized arena for evaluating AI coding agents on real GPU kernel optimization tasks on AMD GPUs.
   :keywords: AgentKernelArena, ROCm, GPU, kernel, optimization, agent, LLM, HIP, Triton, benchmark, AMD

******************************
AgentKernelArena documentation
******************************

AgentKernelArena is a standardized evaluation arena, built by AMD, that measures
how well AI coding agents perform on real GPU kernel optimization tasks. It runs
LLM-powered agents side-by-side on the same tasks in isolated workspaces and
scores them with objective, reproducible metrics for compilation, correctness,
and GPU performance.

The AgentKernelArena source code is hosted in the
`AMD-AGI/AgentKernelArena <https://github.com/AMD-AGI/AgentKernelArena>`_ GitHub
repository.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install AgentKernelArena <install/install>`

   .. grid-item-card:: How to

      * :doc:`Run an evaluation <how-to/run-evaluation>`
      * :doc:`Run tasks in parallel across multiple GPUs <how-to/parallel-run>`
      * :doc:`Configure agents and models <how-to/agents>`
      * :doc:`Add a task <how-to/add-task>`
      * :doc:`Validate tasks <how-to/task-validator>`
      * :doc:`Visualize and compare runs <how-to/visualization>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: Reference

      * :doc:`Configuration and API reference <reference/api-reference>`
      * :doc:`Performance benchmark methodology <reference/benchmark-methodology>`

To contribute to the documentation, see the
`AgentKernelArena GitHub repository <https://github.com/AMD-AGI/AgentKernelArena>`_.

AgentKernelArena is released under the Apache 2.0 license. For details, see the
:doc:`License <about/license>` page.
