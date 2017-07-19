#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report  

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

These are bar charts of the occurrences (counts) of each sign class in each data set

![Occurrences Training][visuals/occurrence_training.png]

![Occurrences Validation][visuals/occurrence_validation.png]

![Occurrences Test][visuals/occurrence_test.png]

Additionally, here is a grid of example sign classes with titles

![Sign Grid][visuals/sign_grid.png]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

To preprocess images, I opted to simply normalize them using cv2.normalize()

If normalization was not enough to get the accuracy necessary, augmentation would have been my next choice.
If I ended up augmented the data, I would have experimented with random rotations and possibly some skewing or transforming on the images.

I did notice many of the images were very dark. Investing some time to brighten images and possibly increase contrast would probably yield a significant accuracy improvement.

I figured it wasn't a bad idea to keep all three RGB channels since color could be important for sign classification i.e. certain signs always have the same color.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution       	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x16 	|
| Convolution   	    | 1x1 stride, valid padding, outputs 10x10x64   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x64   	|
| Fully connected		| 1600 nodes down to 800 nodes        			|
| RELU					|												|
| Dropout				| Dropout with keep probability of 0.5			|
| Fully connected		| 800 nodes down to 200 nodes        			|
| RELU					|												|
| Dropout				| Dropout with keep probability of 0.5			|
| Fully connected		| 200 nodes down to 43 nodes       			    |
| Softmax				| Minimize loss cross softmax cross entropy     |
|						| Using the AdamOptimizer						|


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I bumped the batch size to 512 figuring my computer could handle the extra memory usage.nd;lasjdf
I also increased the number of epochs to 25 after noticing the accuracy was still improving after 10 cycles.
I left other parameters like learning rate and standard deviations, etc. as they were in our project.
I used the Adam Optimizer from the previous lab since it provided everything necessary for this sort of network.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of presumably 100% since it would memorize the training set
* validation set accuracy of a peak at 96.3%, ending at 95.5%
* test set accuracy of 95.4%

I started by implementing the LeNet architecture from the previous lab. This yielded mid 80s in accuracy.
I then added dropouts to the fully connected layers and that increased accuracy by only a little bit.
Next, I tried making the network a bit deeper by switching up the convolution layers to produce a deeper network. This increased the accuracy by a significant amount (up to 90-92%)
After that, I normalized the images to have a mean of 0 and to be between +-1. This got me well over 93%
At this time, I noticed that the accuracy was still climbing after 10 epochs, so I bumped it to 20 then 25. I would like to go higher, but my personal computer takes too long. I will investigate using AWS GPUs in the near future.


###Test a Model on New Images

####1. Choose six German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![70 Speed][extra-images/speed70-4.jpg] ![Priority 12][extra-images/priority-12.jpg] ![Pedestrians 27][extra-images/pedestrians-27.jpg]
![Children 28][extra-images/children-28.jpg] ![Ahead 35][extra-images/ahead-35.jpg] ![Roundabout 40][extra-images/roundabout-40.jpg]

I figured the first image would be tricky because it was slightly skewed.
The second image was possibly difficult due to it's similar blue colors.
I guessed that the third image (pedestrians) would be relatively easy.
I also guessed the children crossing/playing wouldn't be too hard for the network to predict correctly.
The continue ahead sign is quite large relative to the 32px window, so it could prove hard to predict.
Lastly, the roundabout image is extremely skewed and is partially cut off (intentionally)

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 70   		| Speed Limit 70   								|
| Priority     			| Priority 										|
| Pedestrians			| General Caution								|
| Children	      		| Children    					 				|
| Straight Ahead		| Straight Ahead     							|
| Roundabout     		| Roundabout        							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%.
This is below the accuracy of the test set, but given the small sample size of n=6, a correct count of 5 is the highest possible without getting them all correct.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the Speed Limit 70 sign, it was very confident at 90.7%:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .9071         		| Speed Limit (70km/h)                          |
| .0692     			| Speed Limit (30km/h)						    |
| .0071					| Speed Limit (20km/h)					     	|
| .0038	      			| Traffic signals		 			         	|
| .0027				    | General caution				            	|

For the Priority sign, it was extremely confident with a 100% level. All other values were so tiny, they rounded to 0.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.000         		| Priority road                                 |
| 0     			    | Stop						                    |
| 0					    | End of no passing					           	|
| 0	      			    | Road work		 			                	|
| 0				        | Traffic signals                               |

For the Pedestrian sign, it was confident (100%) that it was incorrectly a General caution sign.
These two signs are very similar. It's possible that the small pixel count and downscaling of the image did not help.
It comes as no surprise that the correct sign was under-represented in the overall training data. Augmenting images would have helped this.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0            		| General caution                               |
| 0     			    | Pedestrians				                    |
| 0					    | Road narrows		    			           	|
| 0	      			    | Bicycles crossing			                	|
| 0				        | Traffic signals                               |

For the Children sign, it was fairly confident (almost 80%) but had a significant second guess at 18%:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.7977         		| Children crossing                             |
| 0.1803     		    | General caution				                |
| 0.0174				| Road narrows		    			           	|
| 0.0033     			| Bicycles crossing			                	|
| 0.0008				| Dangerous curve                               |

For the Straight Ahead sign, it again was 100% confident. It is very cool to see it ranking mainly arrow-based signs.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.0            		| Straight ahead                                |
| 0     			    | Go straight or right		                    |
| 0					    | Turn left ahead		    			      	|
| 0	      			    | Go straight or left			               	|
| 0				        | Speed limit 60                                |

For the Roundabout sign, it was almost perfectly confident at 99.97%:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.9997         		| Roundabout                                    |
| 0.0003     		    | End of no passing				                |
| 0			        	| Go straight or left		   		           	|
| 0     	       		| Speed limit 80			                	|
| 0			        	| Turn right ahead                              |

Overall, not bad! There is no surprise that the predictions appear to favor heavily represented signs from the training set (those with many samples). Augmentation of the set to increase less-represented signs would certainly help.