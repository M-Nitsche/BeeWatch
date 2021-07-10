# BeeWatch

## Motivation
(Christin Scheib)  
Insects such as bees, wasps, flies and beetles are the most important pollinators of wild and cultivated plants. From the 109 most important crops such as apples, strawberries, almonds, tomatos and melons,  about 80 % completely depend on the pollination of animals. [1] The total economic value generated by that is estimated to be around 153 billion euros per year worldwide. [2] Thinking of bees, wild bees are often underestimated in their importance when it comes to pollination. While the honey bee is ony responsible for at most one third of the pollination services, the remaining two third is done by wild bees and other wild pollinators such as the bumblebee. [3] However, entomologists worldwide have been oberserving a severe decline in the population of pollinating animals. Mostly affected by this are honey bees as well as wild bees. [4] As the global agriculture sector depends on the pollination of insects the global decline is a severe threat for our ecosystem and the industries. [5] In order to counteract on this trend it is imperative to understand the root causes. Several contibution factors such as harmful pesticides, parasites, diseases, malnutrition, intruders, urbanisation and intensive monoculture cultivation have been found to lead to this global decline. [6, 7, 8, 9, 10] As these are only an subset of all the contributing factors it is hard to understand what caused a local rise in the bee mortality. 






## Dataset

For the data collection different devices were used - a summary can be found in the table below.

| Device        | Camera/Resolution  |
| ------------- |:-------------:|
| IPhone 7      | 12 MP         |  
| IPhone 11 Pro | 12 MP         |
| x             | ...           |


### Labeling
For the initial labeling the [VGG Image Annotator](http://www.robots.ox.ac.uk/~vgg/software/via/via-1.0.6.html) was used. With it's limited export functionalty we quickly ran into problems because we experimented with different models and frameworks which required different formats. This lead us to the holostic object detection platform [Roboflow](https://roboflow.com) which offers several export formats. 

The final Dataset consists of 1.814 images with 104 null examples. Each image has 1.1 annotations on average with results in 2.047 annotation overall with one class [bee]. Images are annotated with bounding boxes.

### Dataset

| Image source  | count         |
| ------------- |:-------------:|
| [Malika Nisal Ratnayake et. al.](https://bridges.monash.edu/articles/dataset/Honeybee_video_tracking_data/12895433)| 436       |  
| own images    | 1398           

Images where eather taken as a single photo, or a video was taken and then deconstructed into single frames. To turn videos into single frames [FFmpeg](https://www.ffmpeg.org) was used. 
Images with various different backgrounds (flowers) are included - and selection of sample images can be seen below.

<p float="left">
  <img src="/doku_resources/image_1.jpg" width="350" />
  <img src="/doku_resources/image_2.jpg" width="350" /> 
  <img src="/doku_resources/image_3.jpg" width="350" />
  <img src="/doku_resources/image_4.jpg" width="350" />
</p>

### Mosaic dataset
The collected image footage is quite limited regarding the diversitiy of different flower types and colors. As the performance of computer vision applications primarly depend on the quality and especially the diversity of the training dataset, we decided to complement the collected data by a more comprehensive and diverse set of images. Therefore we picked the public image and video hosting platform [flickr](https://www.flickr.com) to do a structured search string query. Flickr is due to the extensive supply of word-tagged images from various domains a common and well-known tool for the creation of computer vision datasets. In order for us to comply with data privacy and protection guidelines, we only queried images listed under creative common licence. 
As the quality of the queried images heavily depend on the search string, we evaluated various keywords in advance. The search strings were iteratively evluated by a brief review of the responses and resulted in the following final search string: "bee flowers", "flowers" and "flower bushes". The latter were used for the [synthetic dataset generation](####Synthetic-dataset-generation) as background images.


After labeling bees in the downloaded datasets following the procedure presented in the [labeling section](###labeling) we used them to generate mosaic data. The mosaic augmentation is originally motivated in the YOLOv4 release and contributes significantly to its performance boost ([Bochkovskiy et al., 2020](literature/Bochkovskiy%20et%20al.%20(2020)%20-%20YOLOv4:%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.pdf)). In order to scale down the queried bee images and benefit from the stated performance increase in model implementations beside YOLOv4/5 we generated 1000 mosaic images 3x4 and the corresponding new annotation files (see [mosaic_data_augmentation.ipynb](dataset/mosaic_data_augmentation.ipynb)). The probability of a bee image to be chosen for a individual mosaic tile was set to 0.3. The following shows an example mosaic image:

![example-mosaic-image](doku_resources/mosaic_image.jpg)

### Synthetic dataset generation


## Model

## Deployment
The Jetson Nano is a small powerful device optimized for IoT applications. It comes  with a Quad-core ARM Cortex-A57 MPCore processor, NVIDIA Maxwell architecture with 128 NVIDIA CUDA cores and 4GB RAM. The Jetson Nano does not come with the Linux architecture already setup, instead it is the first step to write the Image file to the SD Card to boot and setup the system. 

### Deployment processes
Having setup the Jetson Nano it was not yet ready for directly detecting bees as we have not deployed the model. For our case three possible deployment options were:(1) docker containers, (2) virtual environments, (3) traditional deployment.  
  
(1) Docker: Docker has several advantages and has become quite popular in the last couple of years.  Maintaining multiple applications is a quite complex process. They are written in different languages and use different frameworks and architectures which makes them difficult to update or to move around. Docker simplifies that by using containers. A container bundles application code with related configuration files, libraries and dependencies. By doing that it can run uniformly and consistently on any infrastructure. Furthermore, it gives developers the freedom to innovate with their choice of tools, application stacks, and deployment environments for each project. Another big advantage of docker containers is that they are portable, so that software can be built locally and then deployed and ran everywhere. Having great benefits also Docker has its downsides. For example, they are consuming much of the host system resources. It will not make applications faster and in the worst case make them slower. Furthermore, the persistent data storage in Docker is more complicated and graphical applications do not work well. Since it is still a new technology the documentation is falling behind and backward compatibility is not guaranteed when updating Docker containers. Despite the benefits of Docker it should not be used to containerize every single application. Docker offers the most advantages to microservices, where an application constitutes of many loosely coupled components.  
Based on the advantages and disadvantages of Docker containers we decided against using it in this course, since we are deploying a single model on which is not consisting of microservices. Furthermore, we are using an edge device with limited system resources and the inference time of our model is of the essence.
(https://www.freecodecamp.org/news/7-cases-when-not-to-use-docker/)
(https://www.infoworld.com/article/3310941/why-you-should-use-docker-and-containers.html) 

(2) Virtual environment 
Virtual environments are a well-known tool for developing code in Python. As previously mentioned every project requires different dependencies. When working with packages in Python, the respecting site packages (third-party libraries) are all stored and retrieved in the same directory. If now two packages need the different versions of the same site-package Python is not able to differentiate between versions in the site-package directory. In order to solve this problem virtual environments are used which creates an isolated environment for python projects. It is considered good practice to create a new virtual environment for every python project as there is no limit to the number of environments. As virtual environments are a lightweight tool to isolate the dependencies of different projects from each other we decided to deploy our model using venv. A module for creating multiple virtual environments where  each has its own Python binary and can have its own independent set of installed Python packages in its respecting site directories. 


https://www.geeksforgeeks.org/python-virtual-environment/
https://docs.python.org/3/library/venv.html

Before starting the deployment process we create a 4 GB swap file. This will avoid that the Jetson Nano becomes unresponsive if its memory gets filled up. The Linux RAM is divided into pages. To free up a page of memory it is copied to the preconfigured space on the hard disk, the swap space. This process is important when the system requires more memory than physically available. In this case the less used pages are swapped out and the kernel gives memory to the current application which needs it immediately. Furthermore, an application uses parts of the pages only during its startup phase. By swapping out these pages memory can be given to other applications. The trade-off that comes with swapping is that disks are very slow compared to the memory. The swap file is created with the following commands: 

  $ sudo fallocate -l 4G /var/swapfile  
  $ sudo chmod 600 /var/swapfile  
  $ sudo mkswap /var/swapfile  
  $ sudo swapon /var/swapfile  
  $ sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'  

# References
[1] Klein, A. M., Vaissiere, B. E., Cane, J. H., Steffan-Dewenter, I., Cunningham, S. A., Kremen, C. & Tscharntke, T. (2007): Importance of pollinators in changing landscapes for world crops. Proceedings of the Royal Society B: Biological Sciences, 274, 303-313.  
[2] Gallai, N., Salles, J. M., Settele, J. & Vaissiere, B. E. (2009): Economic valuation of the vulnerability of world agriculture confronted with pollinator decline. Ecological Economy, 68, 810-821.  
[3] Breeze, T. D., Bailey, A. P., Balcombe, K. G. & Potts, S. G. (2011): Pollination services in the UK: How important are honeybees? Agriculture, Ecosystems & Environment, 142, 137-143.
[4] C. A. Hallmann, M. Sorg, E. Jongejans, H. Siepel, N. Hofland, H. Schwan, W. Stenmans, A. Müller, H. Sumser, T. Hörren, et al. More than 75 percent decline over 27 years in total flying insect biomass in protected areas. PloS one, 12(10):e0185809, 2017.    
[5] L. Hein. The economic value of the pollination service, a review across scales. The Open Ecology Journal, 2(1):74– 82, Sept. 2009  
[6] D. L. Cox-Foster, S. Conlan, E. C. Holmes, G. Palacios, J. D. Evans, N. A. Moran, P.-L. Quan, T. Briese, M. Hornig, D. M. Geiser, V. Martinson, D. vanEngelsdorp, A. L. Kalkstein, A. Drysdale, J. Hui, J. Zhai, L. Cui, S. K. Hutchison, J. F. Simons, M. Egholm, J. S. Pettis, andW. I. Lipkin. A metage- nomic survey of microbes in honey bee colony collapse disorder. Science, 318(5848):283–287, 2007.  
[7] M. Henry, M. Beguin, F. Requier, O. Rollin, J.-F. Odoux, P. Aupinel, J. Aptel, S. Tchamitchian, and A. Decourtye. A common pesticide decreases foraging success and survival in honey bees. Science, 336(6079):348–350, 2012.  
[7] F. Nazzi and F. Pennacchio. Honey bee antiviral immune bar- riers as affected by multiple stress factors: A novel paradigm to interpret colony health decline and collapse. Viruses, 10(4), 2018.  
[9] R. J. Paxton. Does infection by nosema ceranae cause colony collapse disorder in honey bees (apis mellifera)? Journal of Apicultural Research, 49(1):80–84, 2010
[10] D. vanEngelsdorp, J. D. Evans, C. Saegerman, C. Mullin, E. Haubruge, B. K. Nguyen, M. Frazier, J. Frazier, D. Cox- Foster, Y. Chen, R. Underwood, D. R. Tarpy, and J. S. Pettis. Colony collapse disorder: A descriptive study. PLOS ONE, 4(8):1–17, 08 2009  

