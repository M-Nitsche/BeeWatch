# BeeWatch

## Motivation
(Christin Scheib)  
Insects such as bees, wasps, flies and beetles are the most important pollinators of wild and cultivated plants. From the 109 most important crops such as apples, strawberries, almonds, tomatos and melons,  about 80 % completely depend on the pollination of animals. [1] The total economic value generated by that is estimated to be around 153 billion euros per year worldwide. [2] Thinking of bees, wild bees are often underestimated in their importance when it comes to pollination. While the honey bee is ony responsible for at most one third of the pollination services, the remaining two third is done by wild bees and other wild pollinators such as the bumblebee. [3] However, entomologists worldwide have been oberserving a severe decline in the population of pollinating animals. Mostly affected by this are honey bees as well as wild bees. [4] As the global agriculture sector depends on the pollination of insects the global decline is a severe threat for our ecosystem and the industries. [5] In order to counteract on this trend it is imperative to understand the root causes. Several contibution factors such as harmful pesticides, parasites, diseases, malnutrition, intruders, urbanisation and intensive monoculture cultivation have been found to lead to this global decline. [6, 7, 8, 9, 10] As these are only an subset of all the contributing factors it is hard to understand what caused a local rise in the bee mortality. 

## Description of the Use Case (Aleksandar Ilievski)
One approach to understand the factors in the rise of bee mortality is to monitor local activities of wild bee communities. This allows to draw conclusions of the current state of their environment and its effects on wild bees. Negative changes in wild bee activity such as a decline in the number of wild bees in an area can bring light to issues regarding the local ecosystem. As mentioned before, these issues can range from natural factors such as weather and climate changes to human factors such as the impact of agriculture and pollution. This can be a starting point for further investigations to find and eliminate factors causing this decline. At the same time, a rise in wild bee activity after implementing actions to eliminate the identified problems is a measure for a successful recovery of a local ecosystem.  

The reason this approach works so well is because wild bees are great bioindicators. There are around 750 different species of wild bees in Central Europe, and they have an average operating radius of around 100-300 meters but can go up to 1500 meters at maximum. Wild bees are also the most efficient pollinators and are oftentimes highly specialized on the pollination of specific plants. Therefore, a species-rich wild bee community is important for maintaining biodiversity in a local landscape. At the same time, wild bees are highly sensitive to their immediate environment which is why negative changes in the local ecosystem will show in their activity.[11] [12]  

Using wild bees for environmental monitoring is not a new concept. In fact, the idea dates back to 1935 and has since been used to detect impacts of pollution on the environment and pass important environmental-friendly regulations. Methods of wild bee monitoring include counting the number of wild bees visiting a flower, catching living bees with nets or with a variety of types of traps or counting dead bees in a beehive. [13][14] While traditional approaches have been executed manually, newer approaches utilize modern technologies to accelerate the monitoring process. Such technologies can include RFID tags, beehive monitoring through built-in sensors or using cameras and image processing. [15] This is where we want to make our contribution: we want to automize the wild bee monitoring process by creating an automated quantitative tracking system using computer vision methods. 

There are existing papers that have already implemented computer vision powered monitoring systems such as Ratnayake, M. N., Dyer, A. G., & Dorin, A. (2021) who have used a combination of a YOLOv2 network and background subtraction to implement a bee tracking system [16]. Their work is used as a starting point for our own bee tracking system. Our tracking system solely focuses on bee tracking i.e., counting the number of bees within a specific time frame. It does not, however, classify different bee species, as they are hard to distinguish. 

Since wild bees are powerful bioindicators, a bee tracking system could be of interest for a broad base of potential customers, such as
-	the manufacturing industry for monitoring the effects of air and soil pollution,
-	farmers for monitoring and regulating the effects of pesticide usage,
-	beekeepers and beekeeping associations for gaining insights into the state of their own bee colonies,
-	regulatory and research institutions for monitoring biodiversity in regions, supporting research projects and passing regulations in case of violations.

## Dataset
### Data Collection (Aleksandar Ilievski)
For the data collection process, pictures and videos were taken by wild bees using different devices. A summary of the devices that were used can be found in the table below. The data collection process proved to be difficult for a multitude of reasons. First, bee activity is highly dependent on the weather. Since the project started in the end of April, the weather in Karlsruhe was very rainy. In May, Karlsruhe had 22 rainy days and in June there were 17 [17]. This meant that there was quite a short time frame for data collection. During the first data collection round there was also a lot of wind which caused movement of plants in the videos. This is particularly problematic for the usage of background subtraction, since it creates a lot of noise in the footage. Second, it is challenging to take pictures of living insects as they are very small and move fast. When there was wind, the bees tended to move even more rapidly. Therefore, taking a clear picture from an acceptable distance took some practice in the beginning. The third problem is when bees are covered by plants. When a big part of the bee body is hidden, it is difficult to spot the bee and thus, not very helpful in training the model.  

| Device        | Camera/Resolution  |
| ------------- |:-------------:|
| iPhone 7      | 12 MP         |  
| iPhone 11 Pro | 12 MP         |
| iPhone 12 Pro | 12 MP         |


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

### Additional data sources
The collected image footage is quite limited regarding the diversitiy of different flower types and colors. As the performance of computer vision applications primarly depend on the quality ("Garbage in, garbage out") and especially the diversity of the training dataset, we decided to complement the collected data by a more comprehensive and diverse set of images. Therefore we first [downloaded](dataset/flickr_dataset_collection.ipynb) and [labeled](###Labeling) an additional batch of 1000 bee images and two videos, which were seperated into individual frames. Moreover, we downloaded 1000 images of flowers or bushes without any bees as these are especially usefull as null images and were used for the proceeding synthetic generation of another 1000 bee images (see [Synthetic dataset generation](####Synthetic-dataset-generation)).
The resulting additional datasets are listed below.

| Image source  | No. of images  |  No. of labels | No. of null images |
| ------------- | ------------- | ------------- | ------------- |
| Flickr - (Mosaic) images| 1000  |  1034 | 0
| Flickr - Video frames | 741 (360 + 381) |  2398 | 167 
| Synthetic images  | 1000     | 7801 | 0

#### Mosaic dataset
In order to collect another 1000 bee images we picked the public image and video hosting platform [flickr](https://www.flickr.com) to do a structured search string query. Flickr is due to the extensive supply of word-tagged images from various domains a common and well-known tool for the creation of computer vision datasets. In order for us to comply with data privacy and protection guidelines, we only queried images listed under creative common licence. 
As the quality of the queried images heavily depend on the search string, we evaluated various keywords in advance. The search strings were iteratively evluated by a brief review of the responses and resulted in the following final search string: "bee flowers", "flowers" and "flower bushes". The latter were used for the [synthetic dataset generation](####Synthetic-dataset-generation) as background images.

After labeling bees in the downloaded datasets following the procedure presented in the [labeling section](###labeling) we used them to generate mosaic data. The mosaic augmentation is originally motivated in the YOLOv4 release and contributes significantly to its performance boost ([Bochkovskiy et al., 2020](literature/Bochkovskiy%20et%20al.%20(2020)%20-%20YOLOv4:%20Optimal%20Speed%20and%20Accuracy%20of%20Object%20Detection.pdf)). As the downloaded images often show only individual bees on one flower the mosaic augmentation also makes sure the data meets our use case requirements to detect bees from further distance. In order to scale down the queried bees and benefit from the stated performance increase in model implementations beside YOLOv4/5 we generated 3x4 mosaic images and the corresponding new annotation files (see [mosaic_data_augmentation.ipynb](dataset/mosaic_data_augmentation.ipynb)). The probability of a bee image to be chosen for a individual mosaic tile was set to 0.3. The following shows an example mosaic image:

![example-mosaic-image](doku_resources/mosaic_image.jpg)

#### Synthetic dataset generation


## Model

## Deployment
(Christin Scheib)
The Jetson Nano is a small powerful device optimized for IoT applications. It comes  with a Quad-core ARM Cortex-A57 MPCore processor, NVIDIA Maxwell architecture with 128 NVIDIA CUDA cores and 4GB RAM. The Jetson Nano does not come with the Linux architecture already setup, instead it is the first step to write the Image file to the SD Card to boot and setup the system. 

### Deployment process
Having setup the Jetson Nano it was not yet ready for directly detecting bees as we have not deployed the model. For our case three possible deployment options were:(1) docker containers, (2) virtual environments, (3) traditional deployment.  
  
(1) Docker  
Docker has several advantages and has become quite popular in the last couple of years.  Maintaining multiple applications is a quite complex process. They are written in different languages and use different frameworks and architectures which makes them difficult to update or to move around. Docker simplifies that by using containers. A container bundles application code with related configuration files, libraries and dependencies. By doing that it can run uniformly and consistently on any infrastructure. Furthermore, it gives developers the freedom to innovate with their choice of tools, application stacks, and deployment environments for each project. Another big advantage of docker containers is that they are portable, so that software can be built locally and then deployed and ran everywhere. Having great benefits also Docker has its downsides. For example, they are consuming much of the host system resources. It will not make applications faster and in the worst case make them slower. Furthermore, the persistent data storage in Docker is more complicated and graphical applications do not work well. Since it is still a new technology the documentation is falling behind and backward compatibility is not guaranteed when updating Docker containers. Despite the benefits of Docker it should not be used to containerize every single application. Docker offers the most advantages to microservices, where an application constitutes of many loosely coupled components.  
Based on the advantages and disadvantages of Docker containers we decided against using it in this course, since we are deploying a single model on which is not consisting of microservices. Furthermore, we are using an edge device with limited system resources and the inference time of our model is of the essence.
(https://www.freecodecamp.org/news/7-cases-when-not-to-use-docker/)
(https://www.infoworld.com/article/3310941/why-you-should-use-docker-and-containers.html) 

(2) Virtual environment  
Virtual environments are a well-known tool for developing code in Python. As previously mentioned every project requires different dependencies. When working with packages in Python, the respecting site packages (third-party libraries) are all stored and retrieved in the same directory. If now two packages need the different versions of the same site-package Python is not able to differentiate between versions in the site-package directory. In order to solve this problem virtual environments are used which creates an isolated environment for python projects. It is considered good practice to create a new virtual environment for every python project as there is no limit to the number of environments. As virtual environments are a lightweight tool to isolate the dependencies of different projects from each other we decided to deploy our model using venv. A module for creating multiple virtual environments where  each has its own Python binary and can have its own independent set of installed Python packages in its respecting site directories. 


https://www.geeksforgeeks.org/python-virtual-environment/
https://docs.python.org/3/library/venv.html

Before starting the deployment process we created a 4 GB swap file. This avoids that the Jetson Nano becomes unresponsive if its memory gets filled up. The Linux RAM is divided into pages. To free up a page of memory it is copied to the preconfigured space on the hard disk, the swap space. This process is important when the system requires more memory than physically available. In this case the less used pages are swapped out and the kernel gives memory to the current application which needs it immediately. Furthermore, an application uses parts of the pages only during its startup phase. By swapping out these pages memory can be given to other applications. The trade-off that comes with swapping is that disks are very slow compared to the memory. The swap file is created with the following commands: 

 ``` 
  $ sudo fallocate -l 4G /var/swapfile  
  $ sudo chmod 600 /var/swapfile  
  $ sudo mkswap /var/swapfile  
  $ sudo swapon /var/swapfile  
  $ sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0" >> /etc/fstab'  
```
  
To ensure that everything is up to date we updated our package list and the installed packages using the commands:
 
 ``` 
  $ sudo apt-get update
  $ sudo apt-get upgrade
 ``` 
After bringing everything up to date we downloaded the package installer pip and installed the venv package for creating our virtual environment by using the following commands: 

 ``` 
  $ sudo apt install python3-pip
  $ sudo apt install -y python3 venv
```
By doing the following command we createda new virtual environment called env and activated it:

 ``` 
  $ python3 -m venv ~/python-envs/env
  $ source ~/python-envs/env/bin/activate
 ``` 

As many packages require the wheel package for installation we installed it using 
   
 ``` 
  $ pip3 install wheel
 ``` 
Now everything was setup so that we are ready to install the required packages for Yolov5. As this is based on the PyTorch framework we needed to install torch and torchvision. Unfortunately, the Jetson Nano architecture does not support the pip install version, which is why we needed to build it from source by doing the following commands:
``` 
PyTorch v1.8.0
  $ wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
  $ sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
  $ pip3 install Cython
  $ pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
  
torchvision v0.9.0
  $ sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
  $ git clone --branch <version> https://github.com/pytorch/vision torchvision   # see below for version of torchvision to download
  $ cd torchvision
  $ export BUILD_VERSION=0.x.0  # where 0.x.0 is the torchvision version  
  $ python3 setup.py install --user
  $ cd ../
``` 
After having checked if the installation process was successful, we downloaded the remaining packages with pip. The required packages are:
```
  matplotlib>=3.2.2
  numpy>=1.18.5
  opencv-python>=4.1.2
  Pillow
  PyYAML>=5.3.1
  scipy>=1.4.1
  torch>=1.7.0
  torchvision>=0.8.1
  tqdm>=4.41.0
  tensorboard>=2.4.1
  seaborn>=0.11.0
  thop 
  pandas
```
After completing the installation process we ran our model. Here we ran into some problems with the installation of torchvision. The model threw the error that there is no version installed which satisfies the requirements. As we did not work on multiple projects on the Jetson Nano we installed the required packages including torch and torchvision in the global site-packages directory outside of the virtual environment in order to delimit the problem with the torchvision installation. Running the model again led to a performance of roughly five frames per second (fps). 
```
  $ python3 detect.py --source /home/beewatch/Downloads/bees_demo1.mp4 --weights best.pt --conf 0.3
```
Please note that this is the performance without tracking. As previously mentioned it is considered good practice to use a virtual environment for every project you work on. However, we could not find the error that led to the torchvision version error.


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
[11] Pfiffner, L., & Müller, A. (2016): Wild bees and pollination. Research Institute of Organic Agriculture (FiBL).
[12] Kevan, P. G. (1999): Pollinators as bioindicators of the state of the environment: species, activity and diversity. Agriculture, Ecosystems & Environment, 74(1-3), 373-393.
[13] Celli, G., & Maccagnani, B. (2003): Honey bees as bioindicators of environmental pollution. Bulletin of Insectology, 56(1), 137-139.
[14] Prendergast, K. S., Menz, M. H., Dixon, K. W., & Bateman, P. W. (2020): The relative performance of sampling methods for native bees: an empirical test and review of the literature. Ecosphere, 11(5), e03076.
[15] Bromenshenk, J. J., Henderson, C. B., Seccomb, R. A., Welch, P. M., Debnam, S. E., & Firth, D. R. (2015): Bees as biosensors: chemosensory ability, honey bee monitoring systems, and emergent sensor technologies derived from the pollinator syndrome. Biosensors, 5(4), 678-711.
[16] Ratnayake, M. N., Dyer, A. G., & Dorin, A. (2021): Tracking individual honeybees among wildflower clusters with computer vision-facilitated pollinator monitoring. Plos one, 16(2), e0239504.
[17] https://www.karlsruhe.de/b3/wetter/meteorologische_werte/extremwerte.de Date of retrieval: 11.07.2021

