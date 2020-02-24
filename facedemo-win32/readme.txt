Reflectance Field Demo

To accompany "Acquiring the Reflectance Field of a Human Face" 
by Paul Debevec, Tim Hawkins, Chris Tchou, Haarm-Pieter Duiker, 
Westley Sarokin, and Mark Sagar. 

Demo software by Chris Tchou and Dan Maas. 

--------------------------------------------------------------------------------

This demo allows you to virtually relight real faces. You can light the subject as if they were in a variety of real lighting environments captured from around the world, or position and adjust your own virtual lights to achieve whatever effect you desire. 

Different subjects are selected by clicking on the thumbnails at the left (additional face datasets may be downloaded individually; see below). Clicking on Environments at the upper right displays a choice of several different environments. The current lighting environment is displayed in a low-resolution panoramic image below the face. Clicking on the yellow arrows to the right or left of this image will make the environment spin continuously around the subject. 

Clicking on Lights at the upper right displays a reference sphere and three virtual lights. These lights may be moved by dragging them around the sphere. When a light is selected (yellow), you may adjust its brightness, color, and diffusion using the controls at bottom right. The panoramic image below the face shows the cumulative lighting environment due to the three virtual lights. 

The program computes a weighted sum of 512 images, each capturing the appearance of the face when illuminated from a particular direction. Performing this computation in real-time requires a DCT-based compression of the data in lighting-direction space. 


Instructions:

Download and unzip the demo into a directory, then run face.exe (Windows) or face (Linux). To add a new face dataset, simply unzip the package (a new subdirectory will be created under faces). We recommend that you not load more than one face dataset per 40MB of RAM in your system.  

Note:

The face demo interface uses OpenGL extensively; for best performance, use a fast and fully-implemented OpenGL system. (we've achieved good results using NVIDIA and Matrox hardware on Windows, and NVIDIA hardware on Linux).

The face demo depends on the version 1.1 release of the SDL library. The Windows
package includes a ready-made DLL; Linux users may have to download SDL from
www.libsdl.org.
