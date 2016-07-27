# Nvpro-pipeline

nvpro-pipeline is a C++ 11 based object-oriented, cross-platform research rendering pipeline with focus on minimum CPU cost and maximum modularization of the steps of a rendering pipeline. It is being developed by the NVIDIA Developer Technology Platform team.

# Features

* A scene graph based on SceniX including loader and saver plug-ins, and algorithms (dp::sg).
* An abstract rendering interface (dp::rix) which supports any combination of vertex and parameter updates of the following techniques:
  * Vertex binding techniques with and without bindless via
      * Vertex Buffer Objects (VBO OpenGL 2.1)
      * Vertex Array Objetcs (VAO OpenGL 3.0)
      * Vertex Attribute Binding (VAB OpenGL 4.3). This is the preferred technique.
  * Parameter updates including transparent handling of
      * Uniforms
      * Uniform Buffer Objects (UBO) also with bindless
      * Shader Storage Buffer Objects (SSBO)
* A cross-bar (dp::sg::xbar) which translates the scene graph to the RiX rendering API with focus on graph traversal elimination.
* An effect system (dp::fx) which abstracts shaders with focus on code generation for the parameter interface, so that a shader can be written once and used with parameters in uniforms, UBOs, or SSBOs
* An abstract frustum culling module with support for CPU and OpenGL using compute shaders.
* A math library for matrix, vector, and quaternion operations.
* Integration widgets for Qt5 and FreeGLUT 2.x and higher.
* User interface manipulators to navigate through the scene.
* Several example applications including a Viewer project which demonstrates a lot of the features provided by the nvpro-pipeline.

This pipeline is the base of several talks held in the past:

* OpenGL 4.4 Scene Rendering Techniques
  * [Video](http://on-demand.gputechconf.com/gtc/2014/video/S4379-opengl-44-scene-rendering-techniques.mp4)
  * [PDF](http://on-demand.gputechconf.com/gtc/2014/presentations/S4379-opengl-44-scene-rendering-techniques.pdf)
* Advanced Scenegraph Rendering Pipeline
  * [Video](http://on-demand.gputechconf.com/gtc/2013/video/S3032-Advanced-Scenegraph-Rendering-Pipeline.mp4)
  * [PDF](http://on-demand.gputechconf.com/gtc/2013/presentations/S3032-Advanced-Scenegraph-Rendering-Pipeline.pdf)
* Nvpro-Pipeline: A Research Rendering Pipeline
  * [Video](http://on-demand.gputechconf.com/gtc/2015/video/S5148.html)
  * [PDF](http://on-demand.gputechconf.com/gtc/2015/presentation/S5148-Markus-Tavenrath.pdf)

# OS Requirements
This project is mainly developed using NVIDIA graphics hardware on Windows 7 x64 and higher. Linux (Ubuntu 14.04 x64 and higher) and L4T on the [Jetson TK1](http://www.nvidia.com/object/jetson-tk1-embedded-dev-kit.html) board is tested in regular intervals. Other configrations might work, but are untested.

# Build requirements

* Windows: Visual Studio 2013 or Visual Studio 2015.
* Linux: gcc 4.8 or higher, older versions untested.


# Building

* For Windows
  * Clone the repository.
  * Execute git submodule update --init --recursive to get some 3rdparty dependencies.
  * Launch the Visual Studio Desktop Command line for the compiler and architecture you want to build.
  * Launch 3rdPartyBuild.cmd. This will download and/or build all required dependencies. This might take some time.
  * Optional: Install Qt from http://qt.io. The build system will detect the Qt version installed through the registry keys created by the installer of the qt-project.
  * Create a "builds/<your_compiler_version>" folder below the nvpro-pipeline folder, e.g. for Visual Studio 2013 "builds/vc12-x64". The "builds" folder is being ignored by git.
  * Launch the CMake GUI version and choose nvpro-pipeline and source folder and nvpro-pipeline/builds/vc12-x64 as binary folder. CMake 3.0 or higher is required.
  * Press 'Configure' and 'Generate'.
  * Open the solution generated in the builds folder and rebuild all.
* For Linux
  * Clone the repository.
  * Execute git submodule update --init --recursive to get some 3rdparty dependencies.
  * Install devil, glut, and boost system and filesystem libraries.
  * Create a folder named "builds" below the nvpro-pipeline project directory.
  * Change directory to the folder named "builds".
  * run cmake .. to configure.
  * run make -j #jobs to compile.

# Providing Pull Requests

NVIDIA is happy to review and consider pull requests for merging into the main tree of the nvpro-pipeline for bug fixes and features. Before providing a pull request to NVIDIA, please note the following:

* A pull request provided to this repo by a developer constitutes permission from the developer for NVIDIA to merge the provided changes or any NVIDIA modified version of these changes to the repo. NVIDIA may remove or change the code at any time and in any way deemed appropriate.
* Not all pull requests can be or will be accepted. NVIDIA will close pull requests that it does not intend to merge.
* The modified files and any new files must include the unmodified NVIDIA copyright header seen at the top of all shipping files.

# Related Projects
Nvpro-pipeline is a big project glueing together a lot of techniques to reduce CPU cost of rendering a scene graph. If you're interested in smaller samples 
which focus on single topics or GPU algorithms you might want to have a look at [nvpro-samples](http://github.com/nvpro-samples).
